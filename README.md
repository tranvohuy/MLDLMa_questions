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



* Methods for unsupervised-learning

- k-mean: given a set of points (in the space), and an positive integer $k$. We want to decompose points into $k$ sets. Assumption: if two points are close, they have some 'relation'. Therefore, we should group points such that points in a group are close together. What is the method?
 - Pick $k$ random points. Think these are "centered", average masses $m_i$. Run through points in the set, and see if points are closest to which masses. Declare that the points belong to corresponding groups. 
 - Recompute the average masses of each group.
 - Run through points in the set, reassign groups again.
 - Repeat, until "nothing new"
 
 Questions: 
  - How do we pick initial points? We want them to spread out.
  - When do we stop? Either a fixed number of steps, or nothing new happens.
  
- Gaussian mixture-models?:
  - We assume that points follow a mixture-Gaussian with some parameters. 
  - We have to approximate these parameters.
  
