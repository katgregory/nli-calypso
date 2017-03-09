# Some notes from OH on 3/8
# Specific to PA4 - but some things (ex reusing weights for the encoders) might be useful

def encode_w_attn(self, inputs, masks, prev_states, scope="", reuse=False):
	self.atte_cell = AttnGRUCell(self.size, prev_states)
	with vs.variable_scope(..., reuse)
			o, _ = dynamic_rnn...

def decode(self, h_q, h_p):
	with var scope = "answer_start":
		a_s = rnn_cell.linear([h_q, h_p], output_size=self.output_size)
	with var scope = "answer_end":
		a_e = rnn_cell.linear([h_q, h_p], output_size=self.output_size)
	return a_s, a_e


p_o, p_h = encoder.encode(self.premise)
h_o, h_h = encoder.encode(self.hypotheses, init_state=q_h, reuse=True)
self.a_s, self.a_e = decoder.decode(p_h, h_h)

def validate:
	valid_cost = 0
	for valid_x, valid_y in valid_dataset:
		valid_cost = self.test(sess, valid_x, valid_y)