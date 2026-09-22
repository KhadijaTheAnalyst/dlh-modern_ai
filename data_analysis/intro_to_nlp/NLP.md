CNN mainly used for image processing while RNN (recurrent neural network ) is used for NLP.


- auto complete
- translation
- Name entity recognition (NER)
- sentiment analysis

Sequence is very important. you cant say you are how instead of how are you?

- Issue1: number of neurons become a problem because there is no fixed size.
- Issue2: too much computation because we need to convert everything in vector
- Issue3: Parameters are not shared.

Types of RNN

- many to many - language translation (RNN)
- many to one - sentiments analysis - music generation - input node is just one


Vanishing gradient
where gradient is very small it is hardly changing anything and there is no learning

Exploding gradient
when individual derivative become large the final derivative is huge and weights change drastically

vanishing gradient problem is prominent in deep learning
due to VG RNN has short term memory
solution for this is GRU and LSTM

LSTM -> Long Short Term Memory
Gated -> Gated Recurrent Unit



