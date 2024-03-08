import tensorflow as tf
import numpy as np

block_size = 32
batch_size = 8
learning_rate = 3e-4
max_iters = 5000
eval_intervals = 200
epochs = 1
n_embd = 384
n_heads = 6
n_layer = 6
dropout = 0.2
global head_count
head_count = 0

#TODO: Re-define Tokenizer
with open('input.txt','r') as txt:
    text = txt.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[c] for c in l])

data = tf.constant(encode(text), dtype=tf.int32)
n = int(0.9*data.shape[0])
train_data = data[:n]
val_data  = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    x = tf.constant([])
    y = tf.constant([])
    ix = tf.random.uniform(shape=(batch_size,), maxval=data.shape[0]-block_size, dtype=tf.int32).numpy()
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1 : i + block_size +1] for i in ix])
    return x,y

#TensorFlow operates under the assumption of "no gradient calculation" by default during inference. So we dont need to set it to no grad.
def estimate_loss(model):
    out = {}
    for split in ['train','val']:
        losses = np.zeros((eval_intervals,), dtype = float)
        for k in range(eval_intervals):
            X, Y = get_batch(split)
            logits, loss = model(X,Y) 
            losses[k] = loss.numpy() #converting Loss Tensor to numpy
        out[split] = losses.mean()
        
    return out

class Head(tf.keras.layers.Layer):
    
    def __init__(self, head_size, isMasked):
        super().__init__()
        self.key = tf.keras.layers.Dense(head_size)
        self.query = tf.keras.layers.Dense(head_size)
        self.value = tf.keras.layers.Dense(head_size)
        self.isMasked = isMasked #defining the masking layer
        # self.mask = tf.ones((block_size, block_size))
        # self.mask = tf.linalg.band_part(self.mask, -1, 0)
        if (self.isMasked):
            self.mask = tf.cast(tf.linalg.band_part(tf.ones((block_size, block_size)), -1, 0), tf.bool) # creating a boolean masking layer.
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, val,key,qry):

        B, T, C = key.shape #Getting shape of input
        k = self.key(key) #passing through key value layer
        q = self.query(qry) #passing through query value layer
        wei = tf.matmul(qry, tf.transpose(key, perm=[0, 2, 1])) * C**-0.5
        if (self.isMasked):
            wei = tf.where(self.mask, wei, -np.inf) # Setting -inf with respect to the boolean mask
        wei = tf.nn.softmax(wei, axis= -1) # Applying softmax to the last dimensions
        wei = self.dropout(wei) # Passing through dropout layer
        #print(wei.shape)
        v = self.value(val) # Passing input through value layer
        #print(v.shape)
        out = tf.matmul(wei, v) # Multiplying value and the key, value matrix
        return out

class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, num_heads, head_size, isMasked):
        super().__init__()
        self.heads = [Head(head_size,isMasked) for _ in range(num_heads)] #Creating multiple heads
        self.proj = tf.keras.layers.Dense(head_size * num_heads) ##Creating a projection vector
        self.dropout = tf.keras.layers.Dropout(dropout)
    
    def call(self, v,k,q):
        out = tf.keras.layers.Concatenate(axis=-1)([h(v,k,q) for h in self.heads]) #Concatenating the heads output
        out = self.dropout(self.proj(out)) #Passing through the projection layer
        return out
    

class FeedForward(tf.keras.layers.Layer):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(n_embd * 4 , activation="relu"),
            tf.keras.layers.Dense(n_embd, activation="relu"),
            tf.keras.layers.Dropout(dropout),
        ])
        
    def call(self,x):
        return self.net(x)


class EncoderBlock(tf.keras.layers.Layer):
    
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size,isMasked = False) #Defining self attention layer
        self.ffwd = FeedForward(n_embd)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, x):
        att_out = self.sa(x, x, x) #Passing through the self attention layer
        x = self.ln1(tf.add(x,self.dropout(att_out))) #Adding the residual connection
        ff_out = self.ffwd(x)
        x = self.ln2(tf.add(x, self.dropout(ff_out))) #Adding the residual connection
        return x 

class DecoderBlock(tf.keras.layers.Layer):
    
    def __init__(self,n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, isMasked= True) #Defining masked self attention layer
        self.ca = MultiHeadAttention(n_heads, head_size, isMasked=False) #Defining cross attention layer
        self.ffwd = FeedForward(n_embd)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.ln3 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, inp):
        x = inp[0]
        enc_out = inp[1]
        att_out = self.sa(x, x, x) #Passing through the self attention layer
        x = self.ln1(tf.add(x, self.dropout(att_out))) .
        att_out = self.ca(enc_out, enc_out, x) #Passing through the cross attention layer
        x = self.ln2(tf.add(x, self.dropout(att_out)))
        ff_out = self.ffwd(x)
        x = self.ln3(tf.add(x, self.dropout(ff_out)))
        return np.array([x, enc_out])


class Transformer(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = tf.keras.layers.Embedding(vocab_size, n_embd) #token embedding
        self.pos_embedding_table = tf.keras.layers.Embedding(block_size, n_embd) #positional embedding
        enc_blocks = [EncoderBlock(n_embd, n_heads=n_heads) for _ in range(n_layer)] 
        self.encoder_blocks = tf.keras.models.Sequential(enc_blocks) #Creating a sequential model for encoder blocks
        dec_blocks = [DecoderBlock(n_embd, n_heads= n_heads) for _ in range(n_layer)]
        self.decoder_blocks = tf.keras.models.Sequential(dec_blocks) #Creating a sequential model for decoder blocks
        self.lmhead = tf.keras.layers.Dense(vocab_size) #Creating a dense layer for language model head (Converting the output embeddings to respective out probs)
        
    def call(self, idx, targets = None):
        B, T= idx.shape
        token_emb = self.token_embedding_table(idx) #Passing through token embedding layer
        pos_emb = self.pos_embedding_table(tf.range(T)) #Passing through positional embedding layer
        x = tf.add(token_emb, pos_emb) #Adding the token and positional embeddings
        encoder_out = self.encoder_blocks(x) #Passing through the encoder blocks
        x, _ = self.decoder_blocks(np.array([x, encoder_out])) #Passing through the decoder blocks
        logits = self.lmhead(x) #Passing through the language model head
                
        if targets == None:
            loss = None
        else:
            if logits != None:
                B, T, C = logits.shape
            logits = tf.reshape(logits,[B*T, C])
            targets = tf.reshape(targets,[B*T])
            loss = tf.keras.losses.SparseCategoricalCrossentropy()(targets, logits)   # Classification beween all possible tokens. Sparse Categorical Entropy.
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] #Getting the last block_size tokens
            #print(idx_cond.shape)
            logits, loss = self(idx_cond) # type: ignore
            if logits is not None:  # Add condition to check if logits is not None
                logits = logits[:, -1, :]
                #print(logits)
                probs = tf.nn.softmax(logits, axis=-1) #Apply softmax along the last dimension
                idx_next = tf.random.categorical(probs, 1) #Multinomial Distrubution and sampling
                idx = tf.concat((idx, idx_next), axis=1) # Concat the newly generated word would be concatenated to the existing idx for creating auto regressive property
                
        return idx
    

model = Transformer()
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate) #Initializing Adam Optimizer


for epochs in range(epochs):
    
    if epochs % eval_intervals == 0:
        losses = estimate_loss(model)
        print(f"step:{epochs} : train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}") #The epoch summary
    
    xb, yb = get_batch('train') #getting inputs
    
    with tf.GradientTape() as tape:
        logits, loss = model(xb,yb) # type: ignore
        grads = tape.gradient(loss, model.trainable_variables) #Getting gradients for model parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
#Testing the Output of the model

idx = tf.constant([block_size*[0]], dtype=tf.int64)
print(decode(model.generate(idx, max_new_tokens=1000)[0].numpy()))