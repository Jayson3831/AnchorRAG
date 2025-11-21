import random, os
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, BitsAndBytesConfig, AutoModelForCausalLM
from typing import Any, Dict, List, Optional, Tuple

class RRAGLlamaConfig(PretrainedConfig):
    model_type = "rragllama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_name_or_path='',
        load_in_8bit=True,
        input_dim=3,
        hidden_size=4096,
        unk_token='<unk>',
        unk_token_id=0,
        similarity_token='<R>',
        freeze_llm=False,
        num_k=10,
        d_model=256,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.load_in_8bit = load_in_8bit
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.unk_token = unk_token
        self.unk_token_id = unk_token_id
        self.similarity_token = similarity_token
        self.freeze_llm = freeze_llm
        self.num_k = num_k
        self.d_model = d_model
        super().__init__(
            **kwargs,
        )

class RFormer(nn.Module):
    def __init__(self, input_dim, num_k, d_model=256, n_head=4, num_layers=1, dropout=0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_k = num_k
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.input_layer = nn.Linear(in_features=input_dim, out_features=d_model)
        self.position_embeddings = nn.Embedding(num_k, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.attention_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier_layer = nn.Linear(in_features=d_model, out_features=1)
        self.classification_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, x, label=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.input_layer(x)
        position_ids = torch.arange(self.num_k, device=x.device).expand(x.shape[0], self.num_k)
        pe = self.position_embeddings(position_ids)
        x = x + pe
        x = self.attention_layer(x.permute(1, 0, 2)).permute(1, 0, 2)
        y = self.classifier_layer(x)
        if label is not None:
            label = label.to(y.dtype)
        loss = self.classification_loss(y, label) if self.training else None
        return x, loss

class RRAGLlamaForCausalLM(PreTrainedModel):
    config_class = RRAGLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    def __init__(self, config):
        super().__init__(config)
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, 
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            )
        if config.freeze_llm:
            print('freeze_llm')
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        self.r_former = RFormer(input_dim=config.input_dim, num_k=config.num_k, d_model=config.d_model).to(self.llama_model.device)
        self.llama_proj = nn.Linear(config.d_model, config.hidden_size, device=self.llama_model.device)

    def get_input_embeddings(self):
        return self.llama_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llama_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llama_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llama_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.llama_model.set_decoder(decoder)

    def get_decoder(self):
        return self.llama_model.get_decoder()

    def encode_retrieval_data(
        self, 
        embeds: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        ):
        """
        该方法的核心作用是将检索到的文档特征(embeds)通过R-Former模型处理后，生成注入到LLM的词嵌入表示(inject_embeds)。

        执行过程:
            embeds 和 label 被送入 self.r_former。
            R-Former（一个 Transformer Encoder）对 k 个检索文档的特征进行编码，输出上下文感知的表示。
            self.llama_proj 线性层将 R-Former 的输出维度从 d_model 映射到 LLM 的 hidden_size。

        执行结果：
            inject_embeds: 经过 R-Former 处理和维度映射后的增强嵌入，形状为 (batch_size, num_k, hidden_size)。
            loss: R-Former 计算出的分类损失，是一个标量张量。
        """
        if embeds.dim() <= 2 and self.config.input_dim == 1:
            embeds = embeds.unsqueeze(-1)
        if label is not None and label.dim() <= 2:
            label = label.unsqueeze(-1)
        logits, loss = self.r_former(embeds, label)
        inject_embeds = self.llama_proj(logits)
        return inject_embeds, loss

    def encode_inputs(self, 
        input_ids: torch.Tensor, 
        embeds: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ):
        """
        该方法的核心作用是将原始的检索信息(embeds)通过R-Former模型处理后，注入到LLM的输入词嵌入(input_embeds)中。

        input_ids: 输入的 token ID 张量，形状通常为 (batch_size, sequence_length)。
        embeds: 包含检索到的文档特征的张量，形状为 (batch_size, num_k, input_dim)。num_k 是检索到的文档数，input_dim 是每个文档的特征维度。
        label: 检索文档对应的标签（例如，是否相关），形状为 (batch_size, num_k)。
        """
        # 获取LLM的词嵌入层(nn.Embedding)
        embed_tokens = self.get_input_embeddings()
        # 将输入的input_ids转换为对应的词嵌入表示，形状为 (batch_size, sequence_length, embedding_dim)
        input_embeds = embed_tokens(input_ids)

        # 判断是否提供了检索特征embeds，即每个文档的三个得分
        if embeds is not None:
            # 调用encode_retrieval_data方法，使用R-Former模型和MLP处理检索特征embeds，
            # 注入的词嵌入inject_embeds由R-Former输出经过线性投影得到，
            # loss是R-Former计算的二分类损失
            inject_embeds, loss = self.encode_retrieval_data(embeds, label)
            unk_token_id = self.config.unk_token_id
            if input_embeds.dim() == 2:
                raise ValueError('dim error')
            elif input_embeds.dim() == 3:
                updated_input_embeds = input_embeds.clone()

                # replace_idx 是一个二维张量，形状为 (total_unk_tokens, 2)。
                # 每一行 [batch_index, sequence_index] 对应一个 <unk> token 在批处理数据中的位置。
                # .squeeze() 在这里用于处理特殊情况，但通常形状不变
                replace_idx = torch.nonzero(input_ids==unk_token_id).squeeze()

                # 将 inject_embeds 从三维 (batch_size, num_k, hidden_size) 重塑为二维 (batch_size * num_k, hidden_size)。
                # 执行结果: inject_embeds 变为一个向量列表，与 replace_idx 中的索引一一对应。
                inject_embeds = inject_embeds.reshape([-1, inject_embeds.shape[-1]])

                # 使用 replace_idx 提供的索引，将 updated_input_embeds 中 <unk> token 对应的原始词嵌入，替换为 inject_embeds 中由 R-Former 生成的增强嵌入。
                # 执行结果: updated_input_embeds 中所有占位符位置的向量都被替换成了包含丰富检索信息的向量。.to(input_embeds.dtype) 确保了数据类型一致。
                updated_input_embeds[replace_idx[:, 0], replace_idx[:, 1]] = inject_embeds.to(input_embeds.dtype)

                # 返回被修改后的、包含增强信息的词嵌入张量和 R-Former 计算的损失。.contiguous() 确保张量在内存中是连续的，便于后续计算。
                return updated_input_embeds.contiguous(), loss
        else:
            return input_embeds, None

    
    def forward(
        self,
        input_ids: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if embeds is None:
            raise ValueError('embeds is None')
        if label is None:
            raise ValueError('label is None')
        inputs_embeds, loss = self.encode_inputs(input_ids, embeds, label)
        if 'inputs_embeds' in kwargs:
            _ = kwargs.pop('inputs_embeds')
        outputs = self.llama_model(inputs_embeds=inputs_embeds, **kwargs)
        if self.training:
            outputs.loss += loss
        return outputs

    def generate(
            self,
            inputs: Dict[str, torch.Tensor],  # 明确 inputs 是字典
            **kwargs
    ):
        if 'embeds' not in inputs.keys():
            raise ValueError('embeds is None')

        # 核心修改：将 attention_mask 从 inputs 字典中取出，如果存在的话
        attention_mask = inputs.get('attention_mask', None)

        # encode_inputs 仅用于处理 input_ids 和 embeds，不处理 mask
        inputs_embeds, loss = self.encode_inputs(inputs['input_ids'], inputs['embeds'])

        if 'inputs_embeds' in kwargs:
            _ = kwargs.pop('inputs_embeds')

        # 将 attention_mask 传入 generate 方法
        # LLaMA generate 方法签名接受 attention_mask
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,  # <-- 关键！
            **kwargs
        )
        return outputs

    def save_model(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        model_to_save = self.llama_model.module if hasattr(self.llama_model, 'module') else self.llama_model
        if not self.config.freeze_llm:
            model_to_save.save_pretrained(save_directory)
        model_dict = {
            'r_former': self.r_former.state_dict(),
            'llama_proj': self.llama_proj.state_dict()
        }
        torch.save(model_dict, os.path.join(save_directory, 'RRAGLlama_pytorch_model.bin'))

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            raise ValueError("Configuration must be provided with `config` argument.")
        
        model = cls(config)

        # 构建量化配置（只在需要8bit时创建）
        quant_config = None
        if config.load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        llm_path = config.model_name_or_path if config.freeze_llm else pretrained_model_path
        print(f'Load LLM params from: {llm_path}')
        llama_model_class = AutoModelForCausalLM.from_pretrained
        llama_model = llama_model_class(llm_path, device_map="auto", quantization_config=quant_config)
        model.llama_model = llama_model
        
        model_path = os.path.join(pretrained_model_path, 'RRAGLlama_pytorch_model.bin')
        other_model_dict = torch.load(model_path)
        model.r_former.load_state_dict(other_model_dict['r_former'])
        model.llama_proj.load_state_dict(other_model_dict['llama_proj'])
        return model
