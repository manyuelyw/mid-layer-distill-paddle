import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.nn import MSELoss, L1Loss
import random
import json
import numpy
from torch.utils.data import Dataset

import paddle
from paddle.nn import MSELoss


from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoConfig

from torch.optim.lr_scheduler import CosineAnnealingLR

# from utils import save_rank
import numpy as np
import struct
from itertools import accumulate
# torch.set_num_threads(4)

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16
}


def save_rank(log_str, save_path, rank=0):
    with open(save_path, "a") as f:
        f.write(log_str + "\n")


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'


class DistributedMMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        def __init__(self, path):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer,
                dtype=np.int32,
                count=self._len,
                offset=offset)
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, name, rank_number, rank_total, cache=None):

        super().__init__()

        self._path = path
        self._name = name
        self._state = 0
        if cache is not None:
            self._cache = cache
            os.makedirs(self._cache, exist_ok=True)
        else:
            self._cache = None
        self._rank_total = rank_total
        self._rank_number = rank_number
        self._index = None
        self._bin_buffer = None
        self._bin_buffer_mmap = None
        self.max_state, self.history = self._probe_data_path(self._path, self._name, self._rank_total)
        self.total_length = self.history[self.max_state - 1][1]

        self._do_init(self._path, self._name, self._cache, self._state)

    def _probe_data_path(self, path, name, rank_total):

        state = 0
        history = {-1: (0, 0)}
        for state in range(np.iinfo(np.int32).max):
            source_file = path + name + f"_{state}"
            if self.exists(source_file):
                index = self.Index(index_file_path(source_file))
                history[state] = (history[state - 1][1], history[state - 1][1] + len(index))
            else:
                break

        return state, history

    def __getstate__(self):
        return self._path + self._name + "_%d" % (self._state)

    def __setstate__(self, state):
        self._state = state
        self._do_init(self._path, self._name, self._cache, self._state)

    def _do_init(self, path, name, cache, state):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap
        if self._index is not None:
            del self._index

        self._state = state

        source_file = path + name + f"_{self._state}"
        self._index = self.Index(index_file_path(source_file))
        self._bin_buffer_mmap = np.memmap(data_file_path(source_file), mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap
        if self._index is not None:
            del self._index

    def __len__(self):
        return self.total_length

    def _next_file(self):
        self._state += 1
        if self._state >= self.max_state:
            self._state = 0
        # print_rank(f"next_file: {self._state}")
        self._do_init(self._path, self._name, self._cache, self._state)

    def __relative_idx(self, idx):
        res = idx - self.history[self._state][0]
        return res

    def __slice_item(self, start, stop):
        ptr = self._index._pointers[self.__relative_idx(start)]
        sizes = self._index._sizes[self.__relative_idx(start):self.__relative_idx(stop)]
        offsets = list(accumulate(sizes))
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=sum(sizes), offset=ptr)
        return np.split(np_array, offsets[:-1])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            while idx >= self.history[self._state][1] or idx < self.history[self._state][0]:
                self._next_file()
            ptr, size = self._index[self.__relative_idx(idx)]
            return np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
        elif isinstance(idx, slice):
            raise NotImplementedError()

    @property
    def sizes(self):
        return self._index.sizes

    def exists(self, path):
        return (
                os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )


class LMTrainDataset(Dataset):
    def __init__(self, tokenizer, path, split, num, ratio, rng_sample: random.Random):
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.ratio = ratio
        self.max_length = 512
        self.max_prompt_length = 256
        self.rng_sample = rng_sample
        self.lm_ctx = DistributedMMapIndexedDataset(path, f"{split}", 0, 1)

        if os.path.exists(os.path.join(path, f"{split}.jsonl")):
            with open(os.path.join(path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.raw]

        if num == -1:
            self.num = len(self.lm_ctx)
        else:
            self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self._get_lm(index)

    def _get_lm(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)
        return {
            "input_ids": input_ids
        }

    def _process_lm(self, i, samp, model_data, no_model_data, gen_data):
        input_ids = samp["input_ids"]
        source_len = 1

        prompt = None
        if 65535 in input_ids:
            source_len = np.where(input_ids == 65535)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len + 1:]], axis=0)
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len - 1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len - 1] = 1.0
        model_data["position_ids"][i][:input_len - 1] = torch.arange(0, input_len - 1, dtype=torch.long)
        no_model_data["label"][i][:input_len - 1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len - 1] = -100
        no_model_data["loss_mask"][i][:input_len - 1] = 1.0
        no_model_data["loss_mask"][i][:source_len - 1] = 0

        if prompt is not None:
            gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt):] = 1.0

    def collate(self, samples):
        bs = len(samples)

        max_length = self.max_length

        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length),
        }

        model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)

        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }

        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        for i, samp in enumerate(samples):
            self._process_lm(i, samp, model_data, no_model_data, gen_data)

        return model_data, no_model_data, gen_data


def get_teacher_model(teacher_model_path):
    config = AutoConfig.from_pretrained(teacher_model_path)
    model = AutoModelForCausalLM.from_pretrained(teacher_model_path, config=config, torch_dtype=torch.float16)
    model.eval()
    return model


def get_optimizer(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln_f.weight', 'ln_1.weight', 'ln_2.weight', 'ln_cross_attn']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    param_groups = optimizer_grouped_parameters

    # Use AdamW.
    # optimizer = AdamW(param_groups, lr=0.0005, weight_decay=1e-2)
    optimizer = paddle.optimizer.AdamW(learning_rate=0.0005,
                                       parameters=param_groups,
                                       weight_decay=1e-2,)
    return optimizer


def get_learning_rate_scheduler(total_iters, optimizer):
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=0.0001)
    return lr_scheduler


def setup_model_and_optimizer(model_path, total_iters):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16)
    optimizer = get_optimizer(model)
    lr_scheduler = get_learning_rate_scheduler(total_iters, optimizer)
    return model, optimizer, lr_scheduler


def prepare_dataset(data_dir, do_train, do_eval, tokenizer):
    data = {}
    seed = 10
    rng_sample = random.Random(seed)
    if do_train:
        data["train"] = LMTrainDataset(tokenizer, data_dir, "train", 1, 1, rng_sample)
        data["dev"] = LMTrainDataset(tokenizer, data_dir, "valid", 1000, 1, rng_sample)
    elif do_eval:
        data["test"] = LMTrainDataset(tokenizer, data_dir, "valid", 1000, 1, rng_sample)
    else:
        raise ValueError("Do train and do eval must set one")
    return data


# ds_loss
def get_distil_loss(tokenizer, model, teacher_model, model_batch, no_model_batch, logits):
    # 获取teacher各层相应的量输出
    with torch.no_grad():
        teacher_model.eval()

        # 获得teacher model最后一层的输出和teacher_logits
        teacher_outputs = teacher_model(**model_batch, use_cache=False)
        teacher_logits = teacher_outputs.logits
    # 计算教师模型的概率分布
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    # 根据无穷大的logits创建一个掩码
    # inf_mask = torch.isinf(logits)
    inf_mask = paddle.isinf(logits)
    # 计算对数概率分布
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

    # 将教师模型的概率分布与学生模型的对数概率分布相乘得到交叉熵损失。
    prod_probs = paddle.Tensor.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)

    # 根据标签的掩码计算最终的蒸馏损失
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    return distil_loss


# mid_loss
def get_intermediate_distil_loss(tokenizer, model, teacher_model, model_batch, no_model_batch, student_atts, attn_map_network):
    with torch.no_grad():
        teacher_model.eval()

        _, _, teacher_atts = teacher_model(**model_batch, output_attentions=True, output_hidden_states=True,
                                           return_dict=False, use_cache=False)
        teacher_layer_num = len(teacher_atts)
        student_layer_num = len(student_atts)

        teacher_atts = [teacher_att.detach() for teacher_att in teacher_atts]

        max_gcd = int(numpy.gcd(teacher_layer_num, student_layer_num))
        temp_teacher = teacher_layer_num
        temp_student = student_layer_num

        # 若属于情况二：
        i = 0
        while max_gcd == 1:
            if i % 2 == 0:
                temp_teacher -= 1
            else:
                temp_student -= 1
            max_gcd = int(numpy.gcd(temp_teacher, temp_student))
            i += 1
        layers_per_block_teacher = int(temp_teacher / max_gcd)
        layers_per_block_student = int(temp_student / max_gcd)
        map_teacher_atts = [teacher_atts[i * layers_per_block_teacher + layers_per_block_teacher - 1] for i in
                            range(max_gcd)]
        map_student_atts = [student_atts[i * layers_per_block_student + layers_per_block_student - 1] for i in
                            range(max_gcd)]
        if i != 0:
            if i % 2 == 0:
                map_teacher_atts += [teacher_atts[-1]]
                map_student_atts += [student_atts[-1]]
            else:
                map_teacher_atts = map_teacher_atts.pop() + [teacher_atts[-1]]

    # 计算attn_loss
    att_loss = 0.0
    loss_mse = MSELoss()
    # loss_mae = L1Loss()  # 定义MAE损失
    # loss_cos = CosineSimilarity(dim=-1)  # 定义余弦相似度损失

    # attn loss
    for student_att, teacher_att in zip(map_student_atts, map_teacher_atts):
        # [bsz, head_num, seq_len, seq_len]
        # student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att), student_att)
        student_att = paddle.where(student_att <= -1e2, torch.zeros_like(student_att), student_att)
        # teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att), teacher_att)
        teacher_att = paddle.where(teacher_att <= -1e2, torch.zeros_like(teacher_att), teacher_att)
        att_loss += loss_mse(student_att, attn_map_network(teacher_att.permute(0, 3, 2, 1)).permute(0, 3, 2, 1))
    mid_loss = att_loss

    return mid_loss


# lm_loss
def get_teacher_lm_loss(tokenizer, model, teacher_model, model_batch):
    with torch.no_grad():
        t_gen_out = teacher_model.generate(
            **model_batch,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=512,
            top_k=0,
            top_p=1,
            temperature=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=False)

    full_ids = t_gen_out.sequences

    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]
    labels = paddle.Tensor.masked_fill(labels, mask == 0, -100)
    labels[:, :model_batch["input_ids"].size(1) - 1] = -100
    loss_mask = (labels != -100).float()

    new_batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
    }

    # position_ids = torch.cumsum(mask, dim=-1) - 1
    position_ids = paddle.cumsum(mask, axis=-1) - 1
    position_ids = paddle.Tensor.masked_fill(position_ids, mask == 0, 0)
    new_batch["position_ids"] = position_ids

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    outputs = model(**new_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return lm_loss


def finetune(tokenizer: AutoTokenizer, model, optimizer, lr_scheduler, dataset, total_iters, teacher_model=None):
    loss_func = nn.CrossEntropyLoss()

    # sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=0,
    #                              num_replicas=1)
    sampler = paddle.io.DistributedBatchSampler(dataset["train"],
                                                batch_size=1,
                                                num_replicas=1,
                                                rank=0,
                                                shuffle=True,
                                                drop_last=True)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=2, num_workers=4,
        collate_fn=dataset["train"].collate)

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_intermediate_distil_loss = 0.0, 0.0, 0.0

    teacher_attn_head_num = 25
    student_attn_head_num = 12

    attn_map_network = nn.Linear(teacher_attn_head_num, student_attn_head_num, dtype=torch.float16)
    attn_map_network.weight.requires_grad_(True)
    attn_map_network.bias.requires_grad_(True)

    for epoch in range(10):
        sampler.set_epoch(epoch)

        model.train()
        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_dataloader):

            outputs = model(**model_batch, use_cache=False)
            logits = outputs.logits

            _, _, student_atts = model(**model_batch, output_attentions=True, output_hidden_states=True,
                                       return_dict=False, use_cache=False)

            lm_loss = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))

            # loss = lm_loss + distill_loss 按照一定比例整合
            distil_loss = get_distil_loss(tokenizer, model, teacher_model, model_batch, no_model_batch, logits)
            intermediate_distil_loss = get_intermediate_distil_loss(tokenizer, model, teacher_model, model_batch,
                                                                    no_model_batch, student_atts, attn_map_network)
            loss = 0.9 * (0.5 * lm_loss + 0.5 * distil_loss) + 0.1 * intermediate_distil_loss

            model.backward(loss)
            model.step()

            global_loss = loss.item()

            global_distil_loss = 0
            global_intermediate_distil_loss = 0
            if teacher_model is not None:
                global_distil_loss = distil_loss.item()
                total_distil_loss += global_distil_loss

                global_intermediate_distil_loss = intermediate_distil_loss.item()
                total_intermediate_distil_loss += global_intermediate_distil_loss

            total_loss += global_loss

            # Logging
            def get_log(log_loss, log_distil_loss):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f}".format(
                    epoch,
                    step,
                    total_iters,
                    global_step,
                    total_iters,
                    log_loss,
                    log_distil_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                )

            if global_step % 4 == 0 and step % 1 == 0:
                log_str = get_log(
                    total_loss / 4,
                    total_distil_loss / 4)
                save_rank(log_str, os.path.join(".", "log.txt"))
                total_loss, total_distil_loss, total_intermediate_distil_loss = 0.0, 0.0, 0.0

            # Checkpointing
            if global_step % 1 == 0 and step % 1 == 0:
                save_dir_path = os.path.join(".", str(global_step))
                os.makedirs(save_dir_path, exist_ok=True)
                tokenizer.save_pretrained(save_dir_path)
                model.module.save_pretrained(save_dir_path)

            # Evaluation
            # if global_step % 1 == 0 and step % 1 == 0:
            #     evaluate(tokenizer, model, dataset["dev"], "dev", epoch)

            model.train()

            step += 1
            if step % 1 == 0:
                global_step += 1

            if global_step > total_iters:
                break

    return model


# def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device):
#     collate_fn = dataset.collate
#
#     if args.model_parallel:
#         dp_world_size = mpu.get_data_parallel_world_size()
#         dp_rank = mpu.get_data_parallel_rank()
#         dp_group = mpu.get_data_parallel_group()
#         loss_func = mpu.parallel_cross_entropy
#     else:
#         dp_world_size = dist.get_world_size()
#         dp_rank = dist.get_rank()
#         dp_group = None
#         loss_func = nn.CrossEntropyLoss()
#
#     print_rank("\n*****dp size", dp_world_size)
#
#     generation_config = GenerationConfig(
#         do_sample=args.do_sample,
#         top_p=args.top_p,
#         top_k=args.top_k,
#         temperature=args.temperature,
#         repetition_penalty=args.repetition_penalty,
#         max_length=args.max_length,
#         min_length=None,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,
#         return_dict_in_generate=True,
#         output_scores=False
#     )
#
#     sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
#     dataloader = DataLoader(
#         dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
#
#     model.eval()
#     all_loss = 0.0
#     step = 0
#
#     all_response_ids = []
#
#     with torch.no_grad():
#         # model_batch, no_model_batch, gen_data都是dict
#         for it, (model_batch, no_model_batch, gen_data) in enumerate(
#                 tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
#             # dist.barrier()
#             # for rank in range(dist.get_world_size()):
#             #     if dist.get_rank() == rank:
#             #         print(f"rank: {dist.get_rank()}", model_batch["input_ids"][0][:128])
#             #     dist.barrier()
#
#             dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
#
#             logits = model(**model_batch).logits
#             # print_rank("\n*****model_batch.size: ", **model_batch.size())
#             print_rank("\n*****logits.size: ", logits.size())
#             # print_rank("\n*****gen_data.size: ", gen_data.size())
#             # print_rank("\n*****no_model_batch.size: ", no_model_batch.size())
#
#             # lm_losses
#             if args.model_parallel:
#                 lm_losses = loss_func(logits.contiguous().float(), no_model_batch["label"]).view(-1)
#                 loss_mask = no_model_batch["loss_mask"].view(-1)
#                 loss = (lm_losses * loss_mask).sum(-1) / loss_mask.sum(-1)
#             else:
#                 loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
#
#             max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
#             print_rank("\n*****max_new_tokens_length", max_new_tokens)
#
#             if args.eval_gen:
#                 gen_out = model.generate(
#                     **gen_data,
#                     generation_config=generation_config,
#                     max_new_tokens=max_new_tokens)
#
#                 full_ids = gen_out.sequences
#
#                 full_ids = F.pad(
#                     full_ids,
#                     (0, args.max_length - full_ids.shape[1]),
#                     value=tokenizer.pad_token_id,
#                 )
#
#                 response_ids = full_ids[:, gen_data["input_ids"].size(1):]
#                 all_response_ids.append(response_ids)
#
#             dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
#             loss = loss / dp_world_size
#             all_loss += loss.item()
#             step += 1
#
#     if args.eval_gen:
#         all_response_ids = torch.cat(all_response_ids, dim=0)
#         all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
#         all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
#
#         responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
#
#     if get_rank() == 0:
#         if args.eval_gen:
#             references = dataset.answers
#             responses = responses[:len(references)]
#
#             res = compute_metrics(responses, references)
#
#             eval_dir = os.path.join(args.save, "eval", str(epoch))
#             print_rank(eval_dir)
#             os.makedirs(eval_dir, exist_ok=True)
#             with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
#                 for resp in responses:
#                     f.write(json.dumps({"text": resp}) + "\n")
#         else:
#             res = {}
#
#         avg_loss = all_loss / step
#
#         log_str = f"{split} | avg_loss: {avg_loss} | {res}"
#         print_rank(log_str)
#         save_rank(log_str, os.path.join(args.save, "log.txt"))


def main():

    torch.backends.cudnn.enabled = False

    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    model_path = "../minillm/checkpoint/gpt2-base"
    teacher_model_path = "../minillm/result/gpt2-xlarge"
    data_dir = "../minillm/processed_data/dolly/full/gpt2/"
    do_train = True
    do_eval = False
    batch_size = 2
    dp_world_size = 8
    gradient_accumulation_steps = 1
    epochs = 10

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = prepare_dataset(data_dir, do_train, do_eval, tokenizer)

    if do_train:
        train_iters_per_epoch = int(len(dataset["train"]) / (batch_size * dp_world_size * gradient_accumulation_steps))
        total_iters = train_iters_per_epoch * epochs

        save_interval = train_iters_per_epoch
        eval_interval = train_iters_per_epoch

    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_path, total_iters)
    teacher_model = get_teacher_model(teacher_model_path)

    if do_train:
        model = finetune(tokenizer, model, optimizer, lr_scheduler, dataset, total_iters, teacher_model=teacher_model)

    # if do_eval:
    #     evaluate(tokenizer, model, dataset["test"], "test", 0)


if __name__ == "__main__":
    main()
