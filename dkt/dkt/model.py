import torch
import torch.nn as nn

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
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
        self.embedding_head = nn.Embedding(self.args.n_head + 1, self.hidden_dim // 3)  ### 추가한 부분

        # embedding combination projection
        # feature가 4개이므로 *4를 해줌.
        # self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 5, self.hidden_dim)


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
        # test, question, tag, _, mask, interaction = input  # answerCode는 사용하지 않는다.
        test, question, tag, _, i_head, mask, interaction = input  # answerCode는 사용하지 않는다.
        batch_size = interaction.size(0)

        ### Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_head = self.embedding_head(i_head)  ### 추가한 부분

        # 위의 4가지 feature들을 concatenate
        # embed = torch.cat([embed_interaction, embed_test, embed_question, embed_tag,],2,)
        embed = torch.cat([embed_interaction, embed_test, embed_question, embed_tag, embed_head],2,)

        # linear transformation 시켜 사이즈를 줄인다.
        X = self.comb_proj(embed)

        # weight 초기화
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

        ### Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_head = nn.Embedding(self.args.n_head + 1, self.hidden_dim // 3)  ### 추가한 부분

        # embedding combination projection
        # self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 5, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

        ### LSTMattn에 추가된 부분 : Attention (BERT encoder)
        self.config = BertConfig(
            3,  # vocab_size → not used
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

        # test, question, tag, _, mask, interaction, _ = input
        # test, question, tag, _, mask, interaction = input  # answerCode는 사용하지 않음
        test, question, tag, _, i_head, mask, interaction = input  # answerCode는 사용하지 않는다.

        batch_size = interaction.size(0)

        ### Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_head = self.embedding_head(i_head)  ### 추가한 부분

        # embed = torch.cat([embed_interaction,embed_test,embed_question,embed_tag,],2,)
        embed = torch.cat([embed_interaction, embed_test, embed_question, embed_tag, embed_head],2,)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)


        ### LSTMattn에 추가된 부분 : Attention (BERT encoder)
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
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
