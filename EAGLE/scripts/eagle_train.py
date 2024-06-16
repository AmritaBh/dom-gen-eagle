import os
from turtle import shape
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from collections import namedtuple
from typing import Any
from pytorch_metric_learning import losses
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class ProjectionMLP(nn.Module):
    '''
    Code for Projection MLP: edit and clean as needed

    Model to project [CLS] representation onto
    another space, where the contrastive loss will 
    be calculated.
    '''
    def __init__(self):  
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 300))  
        
    def forward(self, input_features):
        x = input_features[:, 0, :]
        return self.layers(x)

class DomainClassifier(nn.Module):
    def __init__(self, dropout=0.1, num_labels=5):
        super(DomainClassifier, self).__init__()
        self.restored = False
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(768, num_labels)
        # self.apply(init_bert_weights)

    def forward(self, x, alpha):
        x = self.dropout(x)
        reverse_x = ReverseLayerF.apply(x, alpha)
        out = self.classifier(reverse_x)
        return out

## SimCLR style contrastive loss

class SimCLRContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        
        try:
            denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        except RuntimeError as e:
            print("DEBUG:")
            print(e)
            # print(self.negatives_mask.shape)
            # print(similarity_matrix.shape)
            # print(self.temperature.shape)
            exit()
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class ContrastivelyInstructedRoberta(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """

    def __init__(self, model: torch.nn.Module, dom: DomainClassifier, mlp: torch.nn.Module, loss_type: str, logger: SummaryWriter, device: str, lambda_w:float) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model
        self.domain_classifier = dom
        self.mlp = mlp
        self.loss_type = loss_type
        self.logger = logger
        self.device = device
        self.lambda_w = lambda_w
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, src_texts:torch.Tensor, src_masks:torch.Tensor, src_texts_perturb:torch.Tensor, src_masks_perturb:torch.Tensor, \
        domain_labels:torch.Tensor, src_labels:torch.Tensor) -> Any:
        """
        Wrap forward call.
        """

        batch_size = src_texts.shape[0]  
        
        # source
        src_output_dic = self.model(src_texts, attention_mask=src_masks, labels=src_labels)
        src_LCE_real, src_logits_real = src_output_dic["loss"], src_output_dic["logits"]

        src_output_dic_perturbed = self.model(src_texts_perturb, attention_mask=src_masks_perturb, labels=src_labels)
        src_LCE_perturb, src_logits_perturb = src_output_dic_perturbed["loss"], src_output_dic_perturbed["logits"]

        # Contrastive loss
        
        if self.loss_type == "simclr":
            ctr_loss = SimCLRContrastiveLoss(batch_size=batch_size)
            ctr_loss.to(self.device)


        if self.loss_type == "simclr":
            src_z_i = self.mlp(src_output_dic["last_hidden_state"])  ## clean
            src_z_j = self.mlp(src_output_dic_perturbed["last_hidden_state"])  ## perturbed
            src_lctr = ctr_loss(src_z_i, src_z_j)
           

            domain_preds = self.domain_classifier(src_output_dic["last_hidden_state"][:, 0], 0.02) ## hard-coded alpha (domain weight. TODO: fix)
            dom_loss = self.criterion(domain_preds, domain_labels)

       

        use_ce_perturb = True  ## change for ablations only
        use_both_ce_losses = True  ## change for ablations only
        use_ctr_loss = True

        if use_ctr_loss:
            if not use_both_ce_losses:
                # loss = self.lambda_w*(src_lctr+tgt_lctr)/2 + lambda_mmd*mmd
                loss = self.lambda_w*dom_loss + (1-self.lambda_w)*src_lctr
            else:
                if use_ce_perturb:
                    loss = (src_LCE_real + src_LCE_perturb)/2 + 0.5*src_lctr\
                          + 1.0*dom_loss # full loss
                    #loss = (src_LCE_real + src_LCE_perturb)/2 + (1-self.lambda_w)*src_lctr
                else:
                    loss = src_LCE_real + (1-self.lambda_w)*src_lctr + self.lambda_w*dom_loss
        else:
            if use_ce_perturb:
                loss = (src_LCE_real + src_LCE_perturb)/2 \
                            + 5.0*dom_loss # full loss
            else:
                loss = src_LCE_real + self.lambda_w*dom_loss

    
        data = {"total_loss":loss, "src_ctr_loss":src_lctr, "dom_loss": dom_loss, "src_ce_loss_real":src_LCE_real,\
            "src_ce_loss_perturb":src_LCE_perturb, "src_logits":src_logits_real}

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  
            data = data_named_tuple(**data)  

        elif isinstance(data, list):
            data = tuple(data)

        return data


## all of the training script:


"""Training code for the detector model"""

import argparse
import subprocess
from itertools import count
from multiprocessing import Process

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
from transformers import *
# from itertools import cycle


import sys
sys.path.insert(0,os.getcwd())

from data_loader import Corpus, EncodedDataset

from utils import summary, distributed

from roberta_cls import RobertaForContrastiveClassification

torch.manual_seed(int(1000))

DISTRIBUTED_FLAG = False  


def setup_distributed(port=29500):
    if not DISTRIBUTED_FLAG:
        return 0, 1

    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return 0, 1

    if 'MPIR_CVAR_CH3_INTERFACE_HOSTNAME' in os.environ:
        from mpi4py import MPI
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        mpi_size = MPI.COMM_WORLD.Get_size()

        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(backend="nccl", world_size=mpi_size, rank=mpi_rank)
        return mpi_rank, mpi_size

    dist.init_process_group(backend="nccl", init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size,
                  max_sequence_length, random_sequence_length):

    real_corpus = Corpus(real_dataset, data_dir=data_dir)
    fake_corpus = Corpus(fake_dataset, data_dir=data_dir)

    real_train, real_valid = real_corpus.train, real_corpus.valid
    real_train_perturb, real_valid_perturb = real_corpus.train_perturb, real_corpus.valid_perturb

    real_domain_labels_train, real_domain_labels_valid = real_corpus.train_domain_labels, real_corpus.valid_domain_labels
    
    fake_train, fake_valid = fake_corpus.train, fake_corpus.valid
    fake_train_perturb, fake_valid_perturb = fake_corpus.train_perturb, fake_corpus.valid_perturb
    fake_domain_labels_train, fake_domain_labels_valid = fake_corpus.train_domain_labels, fake_corpus.valid_domain_labels


    Sampler = DistributedSampler if distributed() and dist.get_world_size() > 1 else RandomSampler

    min_sequence_length = 10 if random_sequence_length else None
    train_dataset = EncodedDataset(real_train, real_train_perturb, real_domain_labels_train, fake_train, fake_train_perturb, fake_domain_labels_train, tokenizer, max_sequence_length, min_sequence_length)
    train_loader = DataLoader(train_dataset, batch_size, sampler=Sampler(train_dataset), num_workers=0, drop_last=True)

    validation_dataset = EncodedDataset(real_valid, real_valid_perturb, real_domain_labels_valid, fake_valid, fake_valid_perturb, fake_domain_labels_valid, tokenizer, max_sequence_length, min_sequence_length)
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=Sampler(validation_dataset))

    return train_loader, validation_loader

def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


def train(model: nn.Module, dom: DomainClassifier, mlp: nn.Module, loss_type: str, optimizer, device: str, src_loader: DataLoader, summary_writer: SummaryWriter, desc='Train', lambda_w=0.5):
    model.train()

    src_train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0
    train_iteration = 0

    # if len(src_loader)==len(tgt_loader):
    #     double_loader = enumerate(zip(src_loader, tgt_loader))
    # elif len(src_loader)<len(tgt_loader):
    #     print("Src smaller than Tgt")
    #     double_loader = enumerate(zip(cycle(src_loader),tgt_loader))
    # else:
    #     double_loader = enumerate(zip(src_loader,cycle(tgt_loader)))
    with tqdm(src_loader, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop:
        torch.cuda.empty_cache()
        for src_texts, src_masks, src_texts_perturb, src_masks_perturb, dom_labels, src_labels in loop:
            
            # src_texts, src_masks, src_texts_perturb, src_masks_perturb, src_labels = src_data[0], src_data[1], src_data[2], src_data[3], src_data[4]
            dom_labels = dom_labels.type(torch.LongTensor)
            src_texts, src_masks, dom_labels, src_labels = src_texts.to(device), src_masks.to(device), dom_labels.to(device), src_labels.to(device)
            src_texts_perturb, src_masks_perturb = src_texts_perturb.to(device), src_masks_perturb.to(device)
            batch_size = src_texts.shape[0]

            optimizer.zero_grad()

            output_dic = model(src_texts, src_masks, src_texts_perturb, src_masks_perturb, dom_labels, src_labels)
            
            loss = output_dic.total_loss

            loss.backward()

            optimizer.step()

            src_batch_accuracy = accuracy_sum(output_dic.src_logits, src_labels)
            src_train_accuracy += src_batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), src_acc=src_train_accuracy / train_epoch_size,\
                     dom_loss=output_dic.dom_loss.item(),\
                    src_LCE_real=output_dic.src_ce_loss_real.item(), src_LCE_perturb=output_dic.src_ce_loss_perturb.item())

    return {
        "train/src_accuracy": src_train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss
    }


def validate(model: nn.Module, device: str, loader: DataLoader, votes=1, desc='Validation'):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}',
                                                               disable=distributed() and dist.get_rank() > 0)]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with tqdm(records, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop, torch.no_grad():
        for example in loop:
            losses = []
            logit_votes = []

            for texts, masks, texts_perturb, masks_perturb, dom_labels, labels in example:
                dom_labels = dom_labels.type(torch.LongTensor) # not using domain labels for validation
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                batch_size = texts.shape[0]

                output_dic = model(texts, attention_mask=masks, labels=labels)
                loss, logits = output_dic["loss"], output_dic["logits"]
                losses.append(loss)
                logit_votes.append(logits)

            loss = torch.stack(losses).mean(dim=0)
            logits = torch.stack(logit_votes).mean(dim=0)

            batch_accuracy = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=validation_accuracy / validation_epoch_size)

    return {
        "validation/accuracy": validation_accuracy,
        "validation/epoch_size": validation_epoch_size,
        "validation/loss": validation_loss
    }


def _all_reduce_dict(d, device):
    # wrap in tensor and use reduce to gpu0 tensor
    output_d = {}
    for (key, value) in sorted(d.items()):
        tensor_input = torch.tensor([[value]]).to(device)
        # torch.distributed.all_reduce(tensor_input)
        output_d[key] = tensor_input.item()
    return output_d


def run(src_data_dir,
        src_real_dataset,
        src_fake_dataset,
        num_sources,
        model_save_path,
        model_save_name,
        batch_size,
        loss_type,
        max_epochs=None,
        device=None,
        max_sequence_length=256,
        random_sequence_length=False,
        epoch_size=None,
        seed=None,
        token_dropout=None,
        large=False,
        learning_rate=2e-5,
        weight_decay=0,
        load_from_checkpoint=False,
        lambda_w=0.5,
        checkpoint_name='',
        **kwargs):
    args = locals()
    rank, world_size = setup_distributed()

    if device is None:
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
       
    #if device=='cpu':
    #    print("Could not find GPU")
    #    exit()

    print('rank:', rank, 'world_size:', world_size, 'device:', device)

    logdir = os.environ.get("OPENAI_LOGDIR", "logs")
    os.makedirs(logdir, exist_ok=True)

    writer = SummaryWriter(logdir) if rank == 0 else None

    import torch.distributed as dist
    if distributed() and rank > 0:
        dist.barrier()

    model_name = 'roberta-large' if large else 'roberta-base'
    tokenization_utils.logger.setLevel('ERROR')
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    roberta_model = RobertaForContrastiveClassification.from_pretrained(model_name).to(device)

    mlp = ProjectionMLP().to(device)
    dom = DomainClassifier(num_labels=num_sources).to(device)
    model = ContrastivelyInstructedRoberta(model=roberta_model, dom=dom, mlp=mlp, loss_type=loss_type, logger=writer, device=device, lambda_w=lambda_w)

    if rank == 0:
        summary(model)
        if distributed():
            dist.barrier()

    if world_size > 1:
        model = DistributedDataParallel(model, [rank], output_device=rank, find_unused_parameters=True)


    src_train_loader, src_validation_loader = load_datasets(src_data_dir, src_real_dataset, src_fake_dataset, tokenizer, batch_size,
                                                    max_sequence_length, random_sequence_length)

    print("length of combined source train data: ")
    print(len(src_train_loader))

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)

    best_validation_accuracy = 0
    without_progress = 0
    earlystop_epochs = 5

    for epoch in epoch_loop:
        if world_size > 1:
            src_train_loader.sampler.set_epoch(epoch)
            src_validation_loader.sampler.set_epoch(epoch)
            

        train_metrics = train(model, dom, mlp, loss_type, optimizer, device, src_train_loader, writer, f'Epoch {epoch}', lambda_w=lambda_w)
        validation_metrics = validate(roberta_model, device, src_validation_loader) ## we are only using supervision on the source

        combined_metrics = _all_reduce_dict({**validation_metrics, **train_metrics}, device)

        combined_metrics["train/src_accuracy"] /= combined_metrics["train/epoch_size"]
        combined_metrics["train/loss"] /= combined_metrics["train/epoch_size"]
        combined_metrics["validation/accuracy"] /= combined_metrics["validation/epoch_size"]
        combined_metrics["validation/loss"] /= combined_metrics["validation/epoch_size"]

        if rank == 0:
            for key, value in combined_metrics.items():
                writer.add_scalar(key, value, global_step=epoch)

            if combined_metrics["validation/accuracy"] > best_validation_accuracy:
                without_progress = 0
                best_validation_accuracy = combined_metrics["validation/accuracy"]

                model_to_save = roberta_model.module if hasattr(roberta_model, 'module') else roberta_model
                torch.save(dict(
                        epoch=epoch,
                        model_state_dict=model_to_save.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        args=args
                    ),
                    os.path.join(model_save_path, model_save_name)
                )

        without_progress += 1

        if without_progress >= earlystop_epochs:
            break


def main(args):

    nproc = int(subprocess.check_output([sys.executable, '-c', "import torch;"
                                         "print(torch.cuda.device_count() if torch.cuda.is_available() else 1)"]))
    nproc=1
    # for machine compatibility
    
    if nproc > 1:
        print(f'Launching {nproc} processes ...', file=sys.stderr)

        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(29500)
        os.environ['WORLD_SIZE'] = str(nproc)
        os.environ['OMP_NUM_THREAD'] = str(1)
        subprocesses = []

        for i in range(nproc):
            os.environ['RANK'] = str(i)
            os.environ['LOCAL_RANK'] = str(i)
            process = Process(target=run, kwargs=vars(args))
            process.start()
            subprocesses.append(process)

        for process in subprocesses:
            process.join()
    else:
        run(**vars(args))
