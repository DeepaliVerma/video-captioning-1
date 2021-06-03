import torch
from torch.utils.data import DataLoader
from config import myConfig
from model.Seq2Seq import Seq2SeqModel
from dataset.VideoCaptionDataset import VideoCaptionDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim
import torch.nn as nn
import os
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Seq2SeqModel(config=myConfig)

trainDataset = VideoCaptionDataset(config=myConfig, json=myConfig.trainJson)
trainDataloader = DataLoader(dataset=trainDataset, batch_size=myConfig.BatchSize, shuffle=True, drop_last=True)

testDataset = VideoCaptionDataset(config=myConfig, json=myConfig.valJson)
testDataloader = DataLoader(dataset=testDataset, batch_size=16, shuffle=False, drop_last=False)

train_size = len(trainDataloader)
test_size = len(testDataloader)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[50, 80], gamma=0.2)

# checkpoint = torch.load('/root/Pycharm_Project/VideoCaption/checkpoint_better/checkpoint_200.pth')
# model.load_state_dict(checkpoint['net'])
model.to(device)

def train(epoch):
    print('====== Train Model ======')
    train_loss=0
    model.train()
    for id, data in enumerate(tqdm(trainDataloader, leave=False, total=train_size)):
        features, decoderInput, decoderTarget = data
        features = features.view(features.size()[0], features.size()[3], -1)
        features = features.to(device)
        decoderInput = decoderInput.to(device)
        decoderTarget = decoderTarget.to(device, dtype=torch.long)
        decoderTarget = decoderTarget.view(-1, )
        optimizer.zero_grad()
        outputs, _ = model(features, decoderInput)
        outputs = outputs.contiguous().view(-1, myConfig.tokenizerOutputdims)
        loss = criterion(outputs, decoderTarget)
        train_loss += loss.item()

        if (id + 1) % 10 == 0:
            print("Train Loss is %.5f" % (train_loss / 10))
            with open("loss.txt", "a") as F:
                F.writelines("{}\n".format(train_loss / 10))
                train_loss = 0
                F.close()

        loss.backward()
        optimizer.step()

    if (epoch+1)%5==0:
        model.eval()
        total=0
        correct=0
        with torch.no_grad():
            print('====== Test Model ======')
            for data in tqdm(testDataloader, leave=False, total=train_size):
                features, decoderInput, decoderTarget = data
                features = features.view(features.size()[0], features.size()[3], -1)
                features = features.to(device)
                decoderInput = decoderInput.to(device)
                decoderTarget = decoderTarget.to(device, dtype=torch.long)
                decoderTarget = decoderTarget.view(-1, )
                outputs,_ = model(features, decoderInput)
                outputs = outputs.contiguous().view(-1, myConfig.tokenizerOutputdims)
                _, predicted = outputs.max(1)
                total += decoderTarget.size(0)
                correct += predicted.eq(decoderTarget).sum().item()

        accuracy=correct/total*100
        print("====== Epoch {} accuracy is {}% ====== ".format(epoch+1,accuracy))

    if (epoch+1)%10==0:
        print('====== Saving model ======')
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_train'):
            os.mkdir('checkpoint_train')
        torch.save(state, './final_checkpoint/checkpoint_%03d.pth'%(epoch+1+200))

def main(epoch):
    a = argparse.ArgumentParser()
    a.add_argument("--debug", "-D", action="store_true")
    a.add_argument("--loss_only", "-L", action="store_true")
    args = a.parse_args()

    dataset = VideoCaptionDataset(config=myConfig, json=myConfig.trainJson)
    vocab = dataset.vocab
    train_data_loader = iter(dataset.train_data_loader)

    decoder = None
    if myConfig.use_recon:
        reconstructor = None
        lambda_recon = torch.autograd.Variable(torch.tensor(1.), requires_grad=True)
        lambda_recon = lambda_recon.to(myConfig.device)

    train_loss = 0
    if myConfig.use_recon:
        train_dec_loss = 0
        train_rec_loss = 0

    forward_reconstructor = None
    for iteration, batch in enumerate(train_data_loader, 1):
        _, encoder_outputs, targets = batch
        encoder_outputs = encoder_outputs.to(myConfig.device)
        targets = targets.to(myConfig.device)
        targets = targets.long()
        target_masks = targets > myConfig.init_word2idx['<PAD>']

        forward_decoder=None
        decoder['model'].train()
        decoder_loss, decoder_hiddens, decoder_output_indices = forward_decoder(decoder, encoder_outputs, targets, target_masks, myConfig.decoder_teacher_forcing_ratio)

        if myConfig.use_recon:
            reconstructor['model'].train()
            recon_loss = forward_reconstructor(decoder_hiddens, encoder_outputs, reconstructor)

        # Loss
        if myConfig.use_recon:
            loss = decoder_loss + lambda_recon * recon_loss
        else:
            loss = decoder_loss

        # Backprop
        decoder['optimizer'].zero_grad()
        if myConfig.use_recon:
            reconstructor['optimizer'].zero_grad()
        loss.backward()
        if myConfig.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(decoder['model'].parameters(), myConfig.gradient_clip)
        decoder['optimizer'].step()
        if myConfig.use_recon:
            reconstructor['optimizer'].step()

            train_dec_loss += decoder_loss.item()
            train_rec_loss += recon_loss.item()
        train_loss += loss.item()

        """ Log Train Progress """
        if args.debug or iteration % myConfig.log_every == 0:
            n_trains = myConfig.log_every * myConfig.batch_size
            train_loss /= n_trains
            if myConfig.use_recon:
                train_dec_loss /= n_trains
                train_rec_loss /= n_trains

            train_loss = 0
            if myConfig.use_recon:
                train_dec_loss = 0
                train_rec_loss = 0

        """ Log Validation Progress """
        if args.debug or iteration % myConfig.validate_every == 0:
            val_loss = 0
            if myConfig.use_recon:
                val_dec_loss = 0
                val_rec_loss = 0
            gt_captions = []
            pd_captions = []
            val_data_loader = iter(dataset.val_data_loader)
            for batch in val_data_loader:
                _, encoder_outputs, targets = batch
                encoder_outputs = encoder_outputs.to(myConfig.device)
                targets = targets.to(myConfig.device)
                targets = targets.long()
                target_masks = targets > myConfig.init_word2idx['<PAD>']

                # Reconstructor
                if myConfig.use_recon:
                    reconstructor['model'].eval()
                    recon_loss = forward_reconstructor(decoder_hiddens, encoder_outputs, reconstructor)

                # Loss
                if myConfig.use_recon:
                    loss = decoder_loss + lambda_recon * recon_loss
                else:
                    loss = decoder_loss

                if myConfig.use_recon:
                    val_dec_loss += decoder_loss.item() * myConfig.batch_size
                    val_rec_loss += recon_loss.item() * myConfig.batch_size
                val_loss += loss.item() * myConfig.batch_size

                _, _, targets = batch
                gt_idxs = targets.cpu().numpy()
                pd_idxs = decoder_output_indices.cpu().numpy()
            n_vals = len(val_data_loader) * myConfig.batch_size
            val_loss /= n_vals
            if myConfig.use_recon:
                val_dec_loss /= n_vals
                val_rec_loss /= n_vals

            msg = "[Validation] Iter {} / {} ({:.1f}%): loss {:.5f}".format(
                iteration, myConfig.n_iterations, iteration / myConfig.n_iterations * 100, val_loss)
            if myConfig.use_recon:
                msg += " (dec {:.5f} + rec {:5f})".format(val_dec_loss, val_rec_loss)
            print(msg)

        if not args.loss_only and (args.debug or iteration % myConfig.test_every == 0):
            pd_vid_caption_pairs = []
            score_data_loader = dataset.score_data_loader
            print("[Test] Iter {} / {} ({:.1f}%)".format(
                iteration, myConfig.n_iterations, iteration / myConfig.n_iterations * 100))
            for search_method in myConfig.search_methods:
                if isinstance(search_method, str):
                    method = search_method
                    search_method_id = search_method
                if isinstance(search_method, tuple):
                    method = search_method[0]
                    search_method_id = "-".join((str(s) for s in search_method))

        """ Save checkpoint """
        if iteration % myConfig.save_every == 0:
            if not os.path.exists(myConfig.save_dpath):
                os.makedirs(myConfig.save_dpath)
            ckpt_fpath = os.path.join(myConfig.save_dpath, "{}_checkpoint.tar".format(iteration))

            torch.save({
                    'iteration': iteration,
                    'dec': decoder['model'].state_dict(),
                    'rec': reconstructor['model'].state_dict(),
                    'dec_opt': decoder['optimizer'].state_dict(),
                    'rec_opt': reconstructor['optimizer'].state_dict(),
                    'loss': loss,
                }, ckpt_fpath)

        if iteration == myConfig.n_iterations:
            break
if __name__=="__main__":
    for i in range(150):
        scheduler.step(i)
        train(i)
