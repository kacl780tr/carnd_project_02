import pickle
import numpy as np
import skimage.transform
import skimage.util
import skimage.io
import skimage.color
import sklearn
import warnings
from pathlib import Path
import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd


def rescale_channel(batch):
	"""
	Rescale all channels of an image to (-1, 1) independently
	param: batch: a 4D array of immages with shape [index, height, width, depth]
	return: rescaled images with data type np.float32
	"""
	batch = batch.astype(np.float32)
	min = np.min(batch, axis=(1,2), keepdims=True)		# not sure this is a good idea, might cause color balance issues
	max = np.max(batch, axis=(1,2), keepdims=True)
	mu = 0.5*(max + min)
	rg = 0.5*(max - min + 1.0e-6)
	batch = (batch - mu)/rg
	return batch


def rescale_flat(batch):
	"""
	Rescale all channels of an image to (-1, 1) simultaneously
	param: batch: a 4D array of immages with shape [index, height, width, depth]
	return: rescaled images with data type np.float32
	"""
	batch = batch.astype(np.float32)
	min = np.min(batch, axis=(1,2,3), keepdims=True)
	max = np.max(batch, axis=(1,2,3), keepdims=True)
	mu = 0.5*(max + min)
	rg = 0.5*(max - min + 1.0e-6)
	batch = (batch - mu)/rg
	return batch


class Data(object):
	"""
	A class for self-contained handling of datasets
	"""
	def __init__(self, feature, label, scaler=None):
		"""
		Initialize
		param: feature: data features, here assumed to be 4d numpy array
		param: label: data labels, assumed to be 1d numpy array
		param: scaler: the scaling function to be applied to generated batches (default = rescale_channel)
		"""
		assert len(feature) == len(label)
		self.feature = feature
		self.label = label
		self.length = len(self.feature)
		self.__data = None
		self.__scaler = scaler
		if scaler is None:
			self.__scaler = rescale_channel

	def classes(self):
		return set(self.label)

	def size_classes(self):
		return len(self.classes())

	def input_shape(self):
		return self.feature[0].shape	# not sure if this will work

	def scale_function(self):
		return self.__scaler

	def shuffle(self):
		self.__data = (self.feature, self.label)
		self.feature, self.label = sklearn.utils.shuffle(self.feature, self.label)

	def reset(self):
		if self.__data:
			self.feature = self.__data[0]
			self.label = self.__data[1]
			self.__data = None

	def batch_count(self, size_batch):
		n = self.length // size_batch
		if self.length % size_batch:
			n += 1
		return n

	def make_batches(self, size_batch):
		for b in range(0, self.length, size_batch):
			x = self.feature[b:b+size_batch]
			y = self.label[b:b+size_batch]
			yield self.__scaler(x), y.astype(np.int32), len(x)

	def make_batch(self):
		return self.__scaler(self.feature), self.label.astype(np.int32), self.length

	def subset(self, n):
		return Data(self.feature[:n], self.label[:n], scaler=self.__scaler)


def load_data(scaler=None, train=True):
	file_train = "./data/train.p"		# These are the pickle files
	file_valid = "./data/valid.p"
	file_test = "./data/test.p"
	if train:
		with open(file_train, mode="rb") as f:
			data_train = pickle.load(f)
		with open(file_valid, mode="rb") as f:
			data_valid = pickle.load(f)
		dataset_train = Data(data_train["features"], data_train["labels"], scaler=scaler)
		dataset_valid = Data(data_valid["features"], data_valid["labels"], scaler=scaler)
		return dataset_train, dataset_valid
	else:
		with open(file_test, mode="rb") as f:
			data_test = pickle.load(f)
		dataset_test = Data(data_test["features"], data_test["labels"], scaler=scaler)
		return dataset_test


def load_augment(filename):
	path = Path(filename)
	if path.exists():
		with open(filename, mode="rb") as f:
			data_augment = pickle.load(f)
		return data_augment
	else:
		return None


def save_augment(filename, dict):
	path = Path(filename)
	if path.exists():
		path.unlink()
	with open(filename, mode="wb") as f:
		pickle.dump(dict, f)


def jitter_image_float(source, xn=0, yn=0):
	"""
	Generate an image shifted by xn, yn pixels
	param: source: input source image
	param: xn: vertical shift in pixels
	param: yn: horizontal shift in pixels
	return: shifted image
	"""
	if xn == 0 and yn == 0:
		return source
	x_pad = [(xn, 0)] if xn > 0 else [(0, -xn)]
	y_pad = [(yn, 0)] if yn > 0 else [(0, -yn)]
	y_pad.append((0,0))
	source_p = skimage.util.pad(source, pad_width=x_pad + y_pad, mode="edge")
	x_crp = [(0, xn)] if xn > 0 else [(-xn, 0)]
	y_crp = [(0, yn)] if yn > 0 else [(-yn, 0)]
	y_crp.append((0,0))
	source_c = skimage.util.crop(source_p, crop_width=x_crp + y_crp)
	return source_c


def transform_image(source, angle=0.0, refine=False, xn=0, yn=0):
	"""
	Generate a slightly modified image from a source image
	param: source: the source image
	param: angle: the rotation angle (optional)
	param: refine: a refinement switch
	param: xn: the desired vertical jitter
	param: yn: the desired horizontal jitter
	return: modified image with shape identical to source
	"""
	if angle == 0.0 and xn == 0 and yn == 0 and not refine:
		return source
	source_f = skimage.util.img_as_float(source)
	if angle != 0.0:
		source_r = skimage.transform.rotate(source_f, angle, order=1, mode="edge")
	else:
		source_r = source_f
	if refine:
		source_u = skimage.transform.pyramid_expand(source_r, order=3)
		source_r = skimage.transform.pyramid_reduce(source_u)
	if xn != 0 or yn != 0:
		source_r = jitter_image_float(source_r, xn=xn, yn=yn)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		return skimage.util.img_as_ubyte(source_r)


def generate_batch(source_image, source_label):
	"""
	Generate a series of images based on stochastic transforms of source images
	param: source_image: the source images assumed to be a 4-d array with axis=0 as the image selection axis
	param: source_label: the source image labels assumed to be a 1-d array
	return: tuple (target_image, target_label)
	note: the labels are not modified in any way, just copied to the output
	"""
	assert len(source_image) == len(source_label)
	bump = np.array([-2, -1, 0, 0, 1, 2], dtype=np.int32)
	generated = np.zeros_like(source_image)
	for i in range(len(source_image)):
		xn = np.random.choice(bump)
		yn = np.random.choice(bump)
		angle = 35.0*np.random.uniform(low=-1.0)
		use_refine = (np.random.uniform() > 0.5)
		if abs(angle) < 2.0:
			angle = 0.0
			use_refine = True
		generated[i,:,:,:] = transform_image(source_image[i], angle, refine=use_refine, xn=xn, yn=yn)
	return generated, source_label


def augment_data(base_data, filename=None):
	"""
	Generate an augmented data set from the input source data
	param: base_data: the input source data set
	param: filename: a list of filenames
	from which previously generated batches will be read,
	or if the files do not exist, to which the generated batches will be written
	return: a dataset containing the union of base data and all generated batches
	"""
	full_data, full_label = None, None
	if filename:
		full_data = base_data.feature
		full_label = base_data.label
		for fn in filename:
			from_file = load_augment(fn)
			if from_file is not None:
				fake_data = from_file["features"]
				fake_label = from_file["labels"]
			else:
				fake_data, fake_label = generate_batch(base_data.feature, base_data.label)
			if from_file is None:
				save_dict = {"features": fake_data, "labels": fake_label}
				save_augment(fn, save_dict)
			full_data = np.concatenate([full_data, fake_data])
			full_label = np.concatenate([full_label, fake_label])
	else:
		fake_data, fake_label = generate_batch(base_data.feature, base_data.label)
		full_data = np.concatenate([base_data.feature, fake_data])
		full_label = np.concatenate([base_data.label, fake_label])
	return Data(full_data, full_label, scaler=base_data.scale_function())


def data_summary(train, valid=None, test=None):
	"""
	Print summary statistics of data sets
	param: train: training data set
	param: valid: validation data set (optional)
	param: test: testing data set (optional)
	return: nothing
	"""
	print("Number of training examples = {}".format(train.length))
	if valid: print("Number of validation examples = {}".format(valid.length))
	if test: print("Number of testing examples = {}".format(test.length))
	print("Image data shape = {}".format(train.input_shape()))
	print("Number of unique classes = {}".format(train.size_classes()))


def data_statistics(datasets):
	"""
	Calculate counts of labels in data sets
	param: datasets: A list of data sets
	return: tuple(classes, list(counts))
	"""
	classes = []
	for ds in datasets:
		classes += ds.classes()
	classes = list(set(classes))
	stats = []
	for ds in datasets:
		counter = Counter(ds.label)
		counts = np.zeros((len(classes)))
		for cls in classes:
			if cls in counter:
				counts[cls] = counter[cls]
		stats.append(counts)
	return (classes, stats)


def plot_statistics(datasets, title=None, setlabels=None):
	"""
	Try to coerce a vertical plot of the frequency counts of data sets
	param: datasets: the data sets for which the plot should be made
	param: title: a plot title (optional)
	param: setlabels: a set of labels for classes (optional)
	return: nothing
	"""
	colors = ["r", "g", "b"]
	classes, stats = data_statistics(datasets)
	n_groups = len(classes)
	fig, axes = plt.subplots(figsize=(20,16))
	index = np.array(classes, dtype=np.float32)
	bar_width = 0.25
	has_legend = bool((setlabels is not None) and len(setlabels) == len(stats))
	idx = 0
	for sts in stats:
		total = np.sum(sts)
		if total > 0.0: sts /= total
		lbl = setlabels[idx] if has_legend else None
		clr = colors[idx % 3]
		plt.bar(index, sts, bar_width, label=lbl, color=clr)
		index += bar_width
		idx += 1
	if title is not None: plt.title(title, fontsize=30)
	plt.xlabel("Traffic sign", fontsize=20)
	plt.ylabel("Fraction of data set", fontsize=20)
	plt.legend()
	plt.tight_layout()
	plt.show()


def plot_statistics_y(datasets, title=None, setlabels=None, figsize=(20,16)):
	"""
	Try to coerce a horizontal plot of the frequency counts of data sets
	param: datasets: the data sets for which the plot should be made
	param: title: a plot title (optional)
	param: setlabels: a set of labels for classes (optional)
	return: nothing
	"""
	colors = ["r", "g", "b"]
	classes, stats = data_statistics(datasets)
	n_groups = len(classes)
	fig, axes = plt.subplots(figsize=figsize)
	index = np.array(classes, dtype=np.float32)
	bar_width = 0.25
	has_legend = bool((setlabels is not None) and len(setlabels) == len(stats))
	idx = 0
	for sts in stats:
		total = np.sum(sts)
		if total > 0.0: sts /= total
		lbl = setlabels[idx] if has_legend else None
		clr = colors[idx % 3]
		plt.barh(index, sts, bar_width, label=lbl, color=clr)
		index += bar_width
		idx += 1
	if title is not None: plt.title(title, fontsize=30)
	names = load_names()
	index -= 2*bar_width
	axes.set_yticks(index)
	axes.set_yticklabels(names)
	axes.invert_yaxis()
	plt.xlabel("Fraction of data set", fontsize=20)
	plt.ylabel("Traffic sign", fontsize=20)
	plt.legend()
	plt.tight_layout()
	plt.show()


def load_names():
	"""
	Read the label names from a csv file
	return: list of strings
	"""
	file_names = "./data/signnames.csv"
	data = pd.read_csv(file_names)
	cols = data.columns
	return data[cols[1]].values


def sample_images(images, labels):
	"""
	Select a sample image from images for each unique label in labels
	param: images: the source image set
	labels: the labels corresponding to the images
	"""
	label_set = list(set(labels))
	selection = []
	for lbl in label_set:
		available = [i for i,lb in enumerate(labels) if lb == lbl]
		k = np.random.choice(available)
		selection.append(images[k])
	return selection


def display_image_samples(dataset):
	"""
	Display a sample of images from a dataset
	param: dataset: the dataset from which to sample
	return: nothing
	"""
	samples = sample_images(dataset.feature, dataset.label)
	subtitles = load_names()
	display_image_grid_alt(samples, title="Image samples", subtitle=subtitles)


def display_image_row(image_set, title=None, subtitle=None):
	"""
	Display a row of images
	Note: untested
	"""
	n = len(image_set)
	fig, axes = plt.subplots(ncols=n)
	fig.tight_layout()
	if title:
		fig.suptitle(title)
	has_subtitles = bool((subtitle is not None) and len(subtitle) == n)
	for j in range(n):
		axes[j].imshow(image_set[j])
		axes[j].set_axis_off()
		if has_subtitles:
			axes[j].set_title(subtitle[j])
	plt.show()


def display_image_grid(image_set, title=None, subtitle=None, ncol=4, figsize=(12,20)):
	"""
	Display a grid of images
	param: image_set: the images to be displayed
	param: title: Title for the plot (optional)
	param: subtitle: Subtitles for each subplot (optional)
	param: ncol: the number of columns to arrange (optional, default = 4)
	param: figsize: the size of the figure (optional)
	return: nothing
	"""
	n = len(image_set)
	nrow = n//ncol
	if n % ncol != 0: nrow += 1
	fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize)
	fig.tight_layout()
	if title:
		fig.suptitle(title, fontsize=20, y=1.0)
	has_subtitles = bool((subtitle is not None) and len(subtitle) == n)
	idx = 0
	for i in range(nrow):
		for j in range(ncol):
			axes[i][j].set_axis_off()
			if idx >= n: continue
			axes[i][j].imshow(image_set[idx])
			if has_subtitles:
				axes[i][j].set_title(subtitle[idx][:20] + "...[" + str(i) + "]")
			idx += 1
	plt.show()


def display_image_grid_alt(image_set, title=None, subtitle=None, ncol=4, figsize=(12,20)):
    """
    Display a grid of images
    param: image_set: the images to be displayed
    param: title: Title for the plot (optional)
    param: subtitle: Subtitles for each subplot (optional)
    param: ncol: the number of columns to arrange (optional, default = 4)
    param: figsize: the size of the figure (optional)
    return: nothing
    """
    # this was written due to problems with the above function in notebooks
    n = len(image_set)
    nrow = n//ncol
    tcks = np.array([])
    if n % ncol != 0: nrow += 1
    fig = plt.figure(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=20, y=0.92)
    has_subtitles = bool((subtitle is not None) and len(subtitle) == n)
    for i in range(n):
        axis = fig.add_subplot(nrow, ncol, i+1)
        plt.xticks(tcks)
        plt.yticks(tcks)
        axis.imshow(image_set[i])
        if has_subtitles:
            axis.set_xlabel(subtitle[i][:20] + "...[" + str(i) + "]")
        else:
            axis.set_xlabel(i)
    plt.show()


def display_predictions(features, actual, predict, figsize=(16, 18)):
	"""
	Display a set of predictions as generated by a trained model
	param: features: the features used to generate the predictions
	param: actual: the actual ground-truth labels
	param: predict: the predictions generated by the model
	return: nothing
	"""
	label_names = load_names()
	n_class = len(label_names)
	n_featr = len(features)
	fig, axes = plt.subplots(nrows=n_featr, ncols=2, figsize=figsize)
	fig.tight_layout()
	fig.suptitle("Softmax predictions", fontsize=20, y=1.1)
	n_predict = len(predict.indices[0])
	bar_index = np.arange(n_predict)
	margin = 0.05
	width = (1.0 - 2.0*margin)/n_predict
	for i, (f, act, pred_idx, pred_val) in enumerate(zip(features, actual, predict.indices, predict.values)):
		name_pred = [label_names[pi] for pi in pred_idx]
		name_actl = label_names[act]
		axes[i][0].imshow(f)
		axes[i][0].set_title(name_actl)
		axes[i][0].set_axis_off()
		axes[i][1].barh(bar_index + margin, pred_val[::-1], width)
		axes[i][1].set_yticks(bar_index + margin)
		axes[i][1].set_yticklabels(name_pred[::-1])
		axes[i][1].set_xticks([0, 0.5, 1.0])
	plt.show()


def prediction_accuracy(actual, predict):
	n = len(actual)
	if n == 0: return 0, 1.0
	correct = 0.0
	for act, prd in zip(actual, predict.indices):
		if act == prd[0]:
			correct += 1.0
	return n, correct/float(n)


def compute_class_scores(confusion):
	"""
	Calculate per-class measures from the the confusion matrix
	param: confusion: the confusion matrix as generated by a model
	return: dictionary of {name: (precision, recall, f-score)}
	"""
	names = load_names()
	n = len(names)
	assert n == confusion.shape[0]
	scores = {}
	for i in range(n):
		nm = names[i]
		true_pos = confusion[i,i]
		flse_pos = np.sum(confusion[:,i]) - true_pos
		flse_neg = np.sum(confusion[i,:]) - true_pos
		prec = true_pos/(true_pos + flse_pos)
		recl = true_pos/(true_pos + flse_neg)
		fscr = 2.0*prec*recl/(prec + recl)
		scores[nm] = (prec, recl, fscr)
	return scores


def read_test_images():
	"""
	Read a set of images from disc and save as pickle file
	Alpha channel information is removed if present
	Images read are assumed to be size 32x32
	return: nothing
	"""
	path = "./data/web/proc"
	files = [f for f in os.listdir(path) if f.startswith("img")]
	imgs_read = np.zeros((len(files), 32, 32, 3), dtype=np.uint8)
	for i in range(imgs_read.shape[0]):
		pth = os.path.join(path, files[i])
		img = skimage.io.imread(pth)
		if img.shape[2] == 4:
			img = skimage.color.rgba2rgb(img)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				img = skimage.util.img_as_ubyte(img)
		imgs_read[i,:,:,:] = img
	labels = np.array([14, 14, 40, 13, 20, 18, 1, 2, 2, 27, 23, 33, 25, 14], dtype=np.int8)
	web_test = Data(imgs_read, labels)
	save_dict = {"features": imgs_read, "labels": labels}
	save_augment("./data/web_test.p", save_dict)

