from loss import *
import torch.optim as optim
from data_processor import *
from trainer import *
import torch.nn.functional as F
from accelerate import Accelerator
class Trainer(object):
    def __init__(
        self,
        model, 
        criterion, 
        optimizer,
        train_loader,
        test_loader_bsds, 
        test_loader_set5, 
        test_loader_set14,
        accelerator):
        self.accelerator =accelerator
        
        self.train_loader=self.accelerator.prepare(train_loader)
        self.test_loader_bsds=test_loader_bsds
        self.test_loader_set5=test_loader_set5
        self.test_loader_set14=test_loader_set14

        self.model=self.accelerator.prepare(model)
        self.criterion=criterion
        self.optimizer=self.accelerator.prepare(optimizer)
        

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return
        state_dict = self.accelerator.get_state_dict(self.model),

        torch.save(state_dict, path)

    def load(self, path):
        accelerator = self.accelerator
        device = accelerator.device
        state_dict = torch.load(path, map_location=device)
        if state_dict.__class__==().__class__:
            state_dict=state_dict[0]
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(state_dict)
    def train_epoch(self,epoch,finetune=False):
        accelerator = self.accelerator
        device = accelerator.device
        accelerator.print('Epoch: %d' % (epoch + 1))
        if finetune:
            self.accelerator.unwrap_model(self.model).encoder.eval()
        else :
            self.model.train()
        sum_loss = 0
        for inputs, _ in self.train_loader:
            inputs = rgb_to_ycbcr(inputs.to(device))[:, 0, :, :].unsqueeze(1) / 255.
            outputs = self.model(inputs)
            loss = self.criterion(outputs[0], inputs) + self.criterion(outputs[1], inputs)
            self.accelerator.backward(loss)

            accelerator.wait_for_everyone()
            self.optimizer.step()
            self.optimizer.zero_grad()
            accelerator.wait_for_everyone()

            sum_loss += loss.item()
        return sum_loss



        
    def valid(self,valid_loader,name):
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:
            sum_psnr = 0
            sum_ssim = 0
            self.model.eval()
            with torch.no_grad():
                for iters, (inputs, _) in enumerate(valid_loader):
                    inputs = rgb_to_ycbcr(inputs.to(device))[:, 0, :, :].unsqueeze(1) / 255.
                    outputs = self.model(inputs)
                    g_inputs=inputs
                    g_outputs=outputs[0]
                    mse = F.mse_loss(g_outputs, g_inputs)
                    psnr = 10 * log10(1 / mse.item())
                    sum_psnr += psnr
                    sum_ssim += ssim(g_outputs, g_inputs)
            PSNR=sum_psnr / len(valid_loader)
            SSIM=sum_ssim / len(valid_loader)

            print(f"----------{name}----------PSNR: {PSNR}----------SSIM: {SSIM}")
        accelerator.wait_for_everyone()
        return
        
