import shutil
from pathlib import Path
import tensorflow as tf
import numpy as np

class Terminator(object):
	def __init__(self, relax=10, mode="ascend"):
		self.__mode_ascend = bool(mode=="ascend")
		self.__relax = int(relax)
		self.__count = int(relax)
		self.__value = None
		self.__viter = None
		self.__iter = int(0)

	def __incr(self):
		self.__iter += 1

	def __ascend(self, value):
		if value > self.__value:
			self.__count = self.__relax
			self.__value = value
			self.__viter = self.__iter
		else:
			self.__count -= 1
			if self.__count == 0: return False
		return True

	def __descend(self, value):
		if value < self.__value:
			self.__count = self.__relax
			self.__value = value
			self.__viter = self.__iter
		else:
			self.__count -= 1
			if self.__count == 0: return False

	def reset(self):
		self.__value = None
		self.__viter = None
		self.__iter = 0
		self.__count = self.__relax

	def new_best(self):
		if self.__value:
			return bool(self.__viter == self.__iter)
		return False

	def current_step(self):
		return self.__viter

	def maintain(self, value):
		self.__incr()
		if self.__value is None:
			self.__value = value
			return True
		if self.__mode_ascend:
			return self.__ascend(value)
		else:
			return self.__descend(value)

	def terminate(self, value):
		return not self.maintain(value)


def reset_dumpdir(dumppath):
	parent = Path(dumppath).parent
	if parent.exists():
		try:
			shutil.rmtree(parent.name)
		except Exception:
			pass
	try:
		parent.mkdir(parents=True, exist_ok=True)
	except Exception:
		pass


def dump_directory(dumppath):
	dir = Path(dumppath)
	if dir.is_dir():
		return dumppath
	else:
		return Path(dumppath).parent.name


train_string = "Epoch {:>3}/{:>3} - Batch {:>3}/{:>3} | Training  : loss = {:>3.5f}, accuracy = {:>3.5f}"
valid_string = "Epoch {:>3}/{:>3} | Validation: loss = {:>3.5f}, accuracy = {:>3.5f}"
test_string = "Testing: loss = {:>3.5f}, accuracy = {:>3.5f}"


def make_measures(logit, label):
	n_logit = logit.shape.as_list()
	with tf.name_scope("measure"):
		label_hot = tf.one_hot(label, n_logit[-1], name="label-hot")
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label_hot, name="x-entropy")
		measure_loss = tf.reduce_mean(cross_entropy, name="loss")
		predict_test = tf.equal(tf.argmax(logit, 1), tf.argmax(label_hot, 1), name="test")
		measure_accuracy = tf.reduce_mean(tf.cast(predict_test, tf.float32), name="accuracy")
	return measure_loss, measure_accuracy


def make_confusion(logit, label):
	n_logit = logit.shape.as_list()
	with tf.name_scope("measure"):
		predict_class = tf.argmax(logit, 1, name="predict")
		measure_confusion = tf.confusion_matrix(label, predict_class, num_classes=n_logit[-1], name="confusion")
	return measure_confusion


def evaluate(session, loss, accuracy, data, size_batch=128):
	g = session.graph
	input = g.get_tensor_by_name("input/image:0")
	label = g.get_tensor_by_name("input/label:0")
	loss_t = 0.0
	accu_t = 0.0
	for dx,dy,n in data.make_batches(size_batch):
		loss_v, accr_v = session.run([loss, accuracy], feed_dict={input:dx, label: dy})
		loss_t += loss_v*n
		accu_t += accr_v*n
	return loss_t/data.length, accu_t/data.length


def evaluate_sum(session, measure, data, size_batch=128):
	g = session.graph
	input = g.get_tensor_by_name("input/image:0")
	label = g.get_tensor_by_name("input/label:0")
	meas_t = np.zeros(measure.shape.as_list(),dtype=np.float32)
	for dx,dy,n in data.make_batches(size_batch):
		meas_v = session.run(measure, feed_dict={input:dx, label: dy})
		meas_t += meas_v
	return meas_t


def test_model(data, dumppath="./checkpoint/model.ckpt", conf_matrix=False):
	dumpdir = dump_directory(dumppath)
	checkpoint = tf.train.latest_checkpoint(dumpdir)
	tf.reset_default_graph()
	with tf.Session() as TS:
		loader = tf.train.import_meta_graph(checkpoint + ".meta")
		loader.restore(TS, checkpoint)
		g = TS.graph
		logit = g.get_tensor_by_name("logit:0")
		label = g.get_tensor_by_name("input/label:0")
		measure_loss, measure_accr = make_measures(logit, label)
		loss, accr = evaluate(TS, measure_loss, measure_accr, data)
		if conf_matrix:
			cmm = make_confusion(logit, label)
			cmv = evaluate_sum(TS, cmm, data)
			return loss, accr, cmv
	return loss, accr


def run_predict(data, dumppath="./checkpoint/model.ckpt", n_top=3):
	dumpdir = dump_directory(dumppath)
	checkpoint = tf.train.latest_checkpoint(dumpdir)
	with tf.Session() as TS:
		loader = tf.train.import_meta_graph(checkpoint + ".meta")
		loader.restore(TS, checkpoint)
		g = TS.graph
		input = g.get_tensor_by_name("input/image:0")
		label = g.get_tensor_by_name("input/label:0")
		logit = g.get_tensor_by_name("logit:0")
		predict = tf.nn.top_k(tf.nn.softmax(logit), n_top)
		X, y, _ = data.make_batch()
		predictions = TS.run(predict, feed_dict={input: X, label: y})
	return predictions

