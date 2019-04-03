# KERAS
- What's wrong with the following code
```
model = Sequential()
model.add(Embedding(input_dim = 1000, output_dim = 64, input_length=10))

input_array = np.random.randn(32, 10)
output_array = model.predict(input_array)
```
- What is the mathematics formula of keras.losses.categorical_crossentropy? input and output of this function?
- Explain the mathematics formula of tensorflow.nn.softmax_cross_entropy_with_logits.
  - What is the "keras" way for tensorflow.nn.softmax_cross_entropy_with_logits"
- Explain keras.losses.binary_crossentropy?
- Explain the mathematics formula of SimpleRNN, LSTM, ...? Compute the number of parameters with given units, and input_shape.

# Math
- We use a Markov Chain to generate sentence. When do we know the sentence will end? Will it runs forever?
- Explain the following code
 ```
 def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
   
 
    probas = np.random.multinomial(1, preds, 1)
       
    return np.argmax(probas)

preds = [0. .1 .3 .4 .2 .3]
for temperature in [0.01 0.1 10]:
  print(sample(preds,temperature))
```
- Given a bunch of points in the plane. How do you know that they come from a Gaussian process, or a Possion point process?
