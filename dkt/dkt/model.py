import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
import math


from transformers.models.bert.modeling_bert import (BertConfig,
                                                        BertEncoder, BertModel)


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        ### Embedding
        # nn.Embedding(num_embeddings, embedding_dim, …)
        # num_embeddings : 해당 카테고리에서 embedding해야 하는 가짓수
        # embedding_dim : embedding했을 때의 vector size
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        # test, question, tag에서 +1한 이유 : train에 정의되지 않은 클래스가 있을 경우를 대비
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_head = nn.Embedding(self.args.n_head + 1, self.hidden_dim // 3)
        self.embedding_hour = nn.Embedding(self.args.n_hour + 1, self.hidden_dim // 3)
        self.embedding_dow = nn.Embedding(self.args.n_dow + 1, self.hidden_dim // 3)

        self.cate_proj = nn.Sequential(nn.Linear((self.hidden_dim//3)*7, self.hidden_dim), nn.LayerNorm(self.hidden_dim))

        self.embedding_cont = nn.Sequential(nn.Linear(self.args.n_cont, self.hidden_dim), nn.LayerNorm(self.hidden_dim))


        self.comb_proj = nn.Sequential(nn.ReLU(),
                        nn.Linear(self.hidden_dim*2, self.hidden_dim),
                        nn.LayerNorm(self.hidden_dim))

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        ### LSTM
        # (순서대로) 입력값의 embedding_dim, hidden state의 embedding_dim, lstm 레이어 수, batch 고려 여부
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)  # Fully connected layer
        self.activation = nn.Sigmoid()


    def init_hidden(self, batch_size):
        # hidden state
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        # cell state
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        batch_size = input["interaction"].size(0)

        # Embedding
        # CATE
        embed_interaction = self.embedding_interaction(input["interaction"])
        embed_test = self.embedding_test(input["testId"])
        embed_question = self.embedding_question(input["assessmentItemID"])
        embed_tag = self.embedding_tag(input["KnowledgeTag"])
        embed_head = self.embedding_head(input["i_head"])
        embed_hour = self.embedding_hour(input["hour"])
        embed_dow = self.embedding_dow(input["dow"])

        embed_cate= torch.cat([
                                embed_interaction, 
                                embed_test, 
                                embed_question, 
                                embed_tag, 
                                embed_head,
                                # embed_mid,
                                # embed_tail,
                                embed_hour,
                                embed_dow,
                                ],2)
        
        embed_cate = self.cate_proj(embed_cate)

        # CONT
        cont = torch.cat([input[bbb].unsqueeze(2) for bbb in self.args.cont_col], 2)
        embed_cont = self.embedding_cont(cont)

        X = self.comb_proj(torch.cat([embed_cate, embed_cont],2))
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)  # 최종 출력과 hidden/cell state를 모두 반환
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)  # batch_size * ? * hidden_dim 

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        ### LSTMattn에 추가된 부분
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_head = nn.Embedding(self.args.n_head + 1, self.hidden_dim // 3)        
        self.embedding_hour = nn.Embedding(self.args.n_hour + 1, self.hidden_dim // 3)
        self.embedding_dow = nn.Embedding(self.args.n_dow + 1, self.hidden_dim // 3)

        self.cate_proj = nn.Sequential(nn.Linear((self.hidden_dim//3)*7, self.hidden_dim), nn.LayerNorm(self.hidden_dim))

        self.comb_proj = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Linear(self.hidden_dim*2, self.hidden_dim),
                            nn.LayerNorm(self.hidden_dim))

        self.embedding_cont = nn.Sequential(nn.Linear(self.args.n_cont, self.hidden_dim), nn.LayerNorm(self.hidden_dim))

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

        self.config = BertConfig(            
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()


    def init_hidden(self, batch_size):
        # hidden state
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        # cell state
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        batch_size = input["interaction"].size(0)
        
        # test, question, tag, _, mask, interaction, _ = input
        # Embedding
        
        embed_interaction = self.embedding_interaction(input["interaction"])
        embed_test = self.embedding_test(input["testId"])
        embed_question = self.embedding_question(input["assessmentItemID"])
        embed_tag = self.embedding_tag(input["KnowledgeTag"])
        embed_head = self.embedding_head(input["i_head"])
        embed_hour = self.embedding_hour(input["hour"])
        embed_dow = self.embedding_dow(input["dow"])

        embed_cate= torch.cat([
            embed_interaction, 
            embed_test, 
            embed_question, 
            embed_tag, 
            embed_head,
            embed_hour,
            embed_dow,
            ],2)
        
        embed_cate = self.cate_proj(embed_cate)

        # cont
        cont = torch.cat([input[bbb].unsqueeze(2) for bbb in self.args.cont_col], 2)
        embed_cont = self.embedding_cont(cont)        

        X = self.comb_proj(torch.cat([embed_cate, embed_cont],2))
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        extended_attention_mask = input['mask'].unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # mask된 부분은 영향력을 매우 작게 한다.
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]  # 가장 마지막 문제를 예측하는 것이므로 -1


        out = self.fc(sequence_output)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_head = nn.Embedding(self.args.n_head + 1, self.hidden_dim // 3)  ### 추가한 부분

        # embedding combination projection
        # self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 5, self.hidden_dim)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        # test, question, tag, _, mask, interaction, _ = input
        # test, question, tag, _, mask, interaction = input
        test, question, tag, _, i_head, mask, interaction = input  # answerCode는 사용하지 않는다.
        batch_size = interaction.size(0)

        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_head = self.embedding_head(i_head)  ### 추가한 부분

        # embed = torch.cat([embed_interaction,embed_test,embed_question,embed_tag,],2,)
        embed = torch.cat([embed_interaction, embed_test, embed_question, embed_tag, embed_head],2,)

        X = self.comb_proj(embed)

        ### Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        

        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds

class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """

    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))


class LastQuery(nn.Module):
    def __init__(self, args):
        super(LastQuery, self).__init__()
        self.args = args
        self.device = args.device        

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_head = nn.Embedding(self.args.n_head + 1, self.hidden_dim // 3)        
        self.embedding_hour = nn.Embedding(self.args.n_hour + 1, self.hidden_dim // 3)
        self.embedding_dow = nn.Embedding(self.args.n_dow + 1, self.hidden_dim // 3)

        self.cate_proj = nn.Sequential(
                            nn.Linear((self.hidden_dim//3)*7, self.hidden_dim), 
                            nn.LayerNorm(self.hidden_dim))
        
        
        self.bn_cont = nn.BatchNorm1d(self.args.n_cont)
        self.embedding_cont = nn.Sequential(
                                nn.Linear(self.args.n_cont, self.hidden_dim), 
                                nn.LayerNorm(self.hidden_dim))

        # embedding combination projection
        self.comb_proj = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Linear(self.hidden_dim*2, self.hidden_dim),
                            nn.LayerNorm(self.hidden_dim))

        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)

        # Encoder
        self.query = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )
        self.key = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )
        self.value = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.args.n_heads
        )
        self.mask = None  # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.args.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)

    def init_hidden(self, batch_size):
        h = torch.zeros(self.args.n_layers, batch_size, self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.args.n_layers, batch_size, self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)
    
    def get_mask(self, seq_len, mask, batch_size):
        new_mask = torch.zeros_like(mask)
        new_mask[mask == 0] = 1
        new_mask[mask != 0] = 0
        mask = new_mask
    
        # batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다
        mask = mask.repeat(1, self.args.n_heads).view(batch_size*self.args.n_heads, -1, seq_len)
        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):
        # Categorical Variable Embedding
        batch_size = input["interaction"].size(0)

        embed_interaction = self.embedding_interaction(input["interaction"])
        embed_question = self.embedding_question(input["assessmentItemID"])
        embed_test = self.embedding_test(input["testId"])        
        embed_tag = self.embedding_tag(input["KnowledgeTag"])
        embed_head = self.embedding_head(input["i_head"])        
        embed_hour = self.embedding_hour(input["hour"])
        embed_dow = self.embedding_dow(input["dow"])

        embed_cate= torch.cat([
                                embed_interaction,             
                                embed_test,        
                                embed_question,      
                                embed_tag, 
                                embed_head,                                
                                embed_hour,
                                embed_dow,
                                ],2)

                
        embed = self.cate_proj(embed_cate)

        # continuous variable embedding
        # batch normalization
        if self.args.n_cont > 0 :
            cont = torch.cat([input[c].unsqueeze(2) for c in self.args.cont_col], 2)
            cont = self.bn_cont(cont.view(-1,cont.size(-1))).view(batch_size,-1,cont.size(-1))
            embed_cont = self.embedding_cont(cont)
            embed = [embed, embed_cont]

        # Running LSTM
        embed = self.comb_proj(torch.cat(embed,2))
        
        ####################### ENCODER #####################
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        self.mask = None
        out, _ = self.attn(q, k, v)

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


class Saint(nn.Module):
    
    def __init__(self, args):
        super(Saint, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.dropout = self.args.dropout
        
        
        ### Embedding 
        # ENCODER embedding
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_head = nn.Embedding(self.args.n_head + 1, self.hidden_dim // 3)
        # self.embedding_mid = nn.Embedding(self.args.n_mid + 1, self.hidden_dim // 3)
        # self.embedding_tail = nn.Embedding(self.args.n_tail + 1, self.hidden_dim // 3)
        self.embedding_hour = nn.Embedding(self.args.n_hour + 1, self.hidden_dim // 3)
        self.embedding_dow = nn.Embedding(self.args.n_dow + 1, self.hidden_dim // 3)

        
        self.bn_cont_e = nn.BatchNorm1d(max(self.args.n_cont_e,1))

        self.embedding_cont_e = nn.Sequential(
                                nn.Linear(max(self.args.n_cont_e,1), self.hidden_dim), 
                                nn.LayerNorm(self.hidden_dim))

        c = min(self.args.n_cont_e,1)
        # encoder combination projection
        self.enc_comb_proj = nn.Sequential(
                            nn.Linear(self.hidden_dim * c+(self.hidden_dim//3)*len(self.args.cate_col_e), 
                                        self.hidden_dim),
                            nn.LayerNorm(self.hidden_dim))

        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_problem_interaction = nn.Embedding(13*2+1, self.hidden_dim//3)
        self.embedding_other = nn.Embedding(self.args.n_other + 1, self.hidden_dim//3)
        self.bn_cont_d = nn.BatchNorm1d(max(self.args.n_cont_d,1))
        self.embedding_cont_d = nn.Sequential(
                                nn.Linear(max(self.args.n_cont_d,1), self.hidden_dim), 
                                nn.LayerNorm(self.hidden_dim))
        # decoder combination projection
        c = min(self.args.n_cont_d,1)
        self.dec_comb_proj = nn.Linear(self.hidden_dim*c+(self.hidden_dim//3)*(len(self.args.cate_col_d)+1), 
                                        self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers, 
            num_decoder_layers=self.args.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.dropout, 
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None
    
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))
        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):
        # 신나는 embedding
        # ENCODER
        be_concat = []
        
        if "testId" in input and "testId" in self.args.cate_col_e:
            embed_test = self.embedding_test(input["testId"])
            be_concat.append(embed_test)

        if "assessmentItemID" in input and "assessmentItemID" in self.args.cate_col_e:
            embed_question = self.embedding_question(input["assessmentItemID"])
            be_concat.append(embed_question)

        if "KnowledgeTag" in input and "KnowledgeTag" in self.args.cate_col_e:
            embed_tag = self.embedding_tag(input["KnowledgeTag"])
            be_concat.append(embed_tag)
            batch_size = input["KnowledgeTag"].size(0)
            seq_len = input["KnowledgeTag"].size(1)

        if "grade" in input and "grade" in self.args.cate_col_e:
            embed_grade = self.embedding_grade(input["grade"])
            be_concat.append(embed_grade)
        
        for c in self.args.cate_col_e :
            if c not in ['assessmentItemID', 'testId', 'KnowledgeTag', 'grade']:
                be_concat.append(self.embedding_other(input[c]))      
            
        if self.args.n_cont_e > 0 :
            cont = torch.cat([input[c].unsqueeze(2) for c in self.args.cont_col_e], 2)
            cont = self.bn_cont_e(cont.view(-1,cont.size(-1))).view(batch_size,-1,cont.size(-1))
            embed_cont_e = self.embedding_cont_e(cont)
            be_concat.append(embed_cont_e)
            
        embed_enc = torch.cat(be_concat, 2)

        embed_enc = self.enc_comb_proj(embed_enc)
        
        # DECODER     
        be_concat = []
        
        if "testId" in input and "testId" in self.args.cate_col_d:
            embed_test = self.embedding_test(input["testId"])
            be_concat.append(embed_test)

        if "assessmentItemID" in input and "assessmentItemID" in self.args.cate_col_d:
            embed_question = self.embedding_question(input["assessmentItemID"])
            be_concat.append(embed_question)

        if "KnowledgeTag" in input and "assessmentItemID" in self.args.cate_col_d:
            embed_tag = self.embedding_tag(input["KnowledgeTag"])
            be_concat.append(embed_tag)

        if "grade" in input and "assessmentItemID" in self.args.cate_col_d:
            embed_grade = self.embedding_grade(input["grade"])
            be_concat.append(embed_grade)
        
        if "interaction" in input :
            embed_interaction = self.embedding_interaction(input["interaction"])
            be_concat.append(embed_interaction)

        if "problem_interaction" in input :
            embed_problem_interaction = self.embedding_problem_interaction(input["problem_interaction"])
            be_concat.append(embed_problem_interaction)

            
        for c in self.args.cate_col_d :
            if c not in ['assessmentItemID', 'testId', 'KnowledgeTag', 'grade']:
                be_concat.append(self.embedding_other(input[c]))
                
        if self.args.n_cont_d > 0 :
            cont = torch.cat([input[c].unsqueeze(2) for c in self.args.cont_col_d], 2)
            cont = self.bn_cont_d(cont.view(-1,cont.size(-1))).view(batch_size,-1,cont.size(-1))
            embed_cont_d = self.embedding_cont_d(cont)
            be_concat.append(embed_cont_d)
            
        embed_dec = torch.cat(be_concat, 2)
        embed_dec = self.dec_comb_proj(embed_dec)
        
        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)
            
        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)
            
        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)
            

        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)
        
        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        mask = input["mask"]
        mask = mask.eq(0)
        
        out = self.transformer(embed_enc, embed_dec,
                            src_mask = self.enc_mask,
                            tgt_mask = self.dec_mask,
                            memory_mask = self.enc_dec_mask,
                            # tgt_mask = self.dec_mask,
                            # src_key_padding_mask = mask,
                            # tgt_key_padding_mask = mask,
                            # memory_key_padding_mask = mask,
                            )

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds