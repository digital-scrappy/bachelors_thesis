
from collections import Counter

def lstm_flop_count(inputs, outputs):
    '''handler for counting the FLOPS of the LSTM cell calculations are based on the pytorch documentation'''

    input_shape =inputs[0].type().sizes()
    output_shape = outputs[0].type().sizes()
    flops = 0

    batch_size = input_shape[1]
    seq_length = input_shape[0]
    input_size = input_shape[2]
    hidden_size = output_shape[2]/2
    w_ih = (4* hidden_size, input_size)
    w_hh = (4* hidden_size, hidden_size)

    flops += w_ih[0]*w_ih[1]
    flops += w_hh[0]*w_hh[1]
    flops += hidden_size*4
    flops += hidden_size*3
    flops += hidden_size*3

    #bidirectional
    flops *= 2

    return Counter({"lstm":int(flops)})


