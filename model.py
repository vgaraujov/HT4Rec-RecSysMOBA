import torch
import torch.nn as nn

class HTransformer(nn.Module):
    def __init__(self, config):
        super(HTransformer, self).__init__()
                
        n_champions = config.dataset.n_champs
        n_items = config.dataset.n_items
        embedding_size = config.model.emb_dim
        nhead = config.model.n_heads
        nlayers1 = config.model.layers1
        nlayers2 = config.model.layers2
        max_seq = config.dataset.max_seq_length
        self.emb_fusion = config.model.emb_fusion
        self.user = config.model.ue
        self.pe = config.model.pe
        self.te = config.model.te

        if self.emb_fusion == 'concat':
            new_embedding_size = int(embedding_size/2)
        else:
            new_embedding_size = embedding_size

        if self.user:
            self.item_embedding = nn.Linear(in_features = n_items, out_features = new_embedding_size) # Linear for multi-hot vector
            self.champion_embedding = nn.Embedding(num_embeddings = n_champions, embedding_dim = new_embedding_size)
            if self.te:
                self.team_embedding = nn.Embedding(num_embeddings = 2, embedding_dim = new_embedding_size)
        else:
            self.item_embedding = nn.Linear(in_features = n_items, out_features = embedding_size) # Linear for multi-hot vector

        if self.pe:
            self.position_embedding = nn.Embedding(num_embeddings = max_seq, embedding_dim = embedding_size)

        encoder_match_layer = nn.TransformerEncoderLayer(d_model = embedding_size, nhead = nhead)
        self.transformer_match_encoder = nn.TransformerEncoder(encoder_match_layer, num_layers = nlayers1)
        
        encoder_seq_layer = nn.TransformerEncoderLayer(d_model = embedding_size, nhead = nhead)
        self.transformer_seq_encoder = nn.TransformerEncoder(encoder_seq_layer, num_layers = nlayers2)
        
        self.out = nn.Linear(embedding_size, n_items)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate_positional_sequence(self, sz, seq_sz):
        position = torch.arange(0, seq_sz, dtype=torch.int64).unsqueeze(0).repeat(sz, 1)
        return position
        
    def generate_team_sequence(self, sz, seq_sz):
        team = torch.ones(seq_sz, dtype=torch.int64)
        team[:5] = 0
        team = team.unsqueeze(0).repeat(sz, 1)
        return team
    
    def forward(self, champion_ids, item_ids): 
        # champion_id is list of champions (B, C)
        # item_ids is list of multihot vectores (B, S, C, I)
        batch_size, seq_size, champ_size, _ = item_ids.size()
        device = item_ids.device
        
        item_emb = self.item_embedding(item_ids) # (B,S,C,H)
        item_emb = item_emb.view(-1,item_emb.size(-2),item_emb.size(-1)) # reshape to (BxS,C,H)

        if self.user:
            champion_emb = self.champion_embedding(champion_ids) # (B,C,H)
            champion_emb = champion_emb.unsqueeze(1).repeat(1, seq_size, 1, 1) # expanding to (B,S,C,H)
            champion_emb = champion_emb.view(-1,champion_emb.size(-2),champion_emb.size(-1)) # (BxS,C,H)
            if self.te:
                team_emb = self.team_embedding(self.generate_team_sequence(batch_size*seq_size, champ_size).to(device)) # (BxS,C,H)
                champion_emb = champion_emb + team_emb
        
            if self.emb_fusion == 'concat':
                match_emb = torch.cat((champion_emb, item_emb), -1) # CONCAT option
            else:
                match_emb = champion_emb + item_emb
        else:
            match_emb = item_emb
        match_emb = match_emb.permute(1,0,2) # (C,BxS,H)

        match_hidden_state = self.transformer_match_encoder(match_emb) # (C,BxS,H)
        user_hidden_state = match_hidden_state[0,:] # get first embedding (similar to [CLS]) # (BxS,H)
        user_hidden_state = user_hidden_state.view(batch_size, seq_size, match_hidden_state.size(-1)) # (B,S,H)
        
        if self.pe:
            position_embedding = self.position_embedding(self.generate_positional_sequence(batch_size, seq_size).to(device)) # (B,S,H)
            seq_hidden_state = user_hidden_state + position_embedding
        else:
            seq_hidden_state = user_hidden_state
        seq_hidden_state = seq_hidden_state.permute(1,0,2) # (S,B,H)

        mask = self.generate_square_subsequent_mask(seq_size).to(device) # (S,S)
        output = self.transformer_seq_encoder(seq_hidden_state, mask) # (S,B,H)
        output = output.permute(1,0,2) # (B,S,H)

        output = output.reshape(-1, output.size(-1)) #(BxS,H)
        
        return self.out(output)
