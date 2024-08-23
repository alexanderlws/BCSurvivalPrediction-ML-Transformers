import random
#import time
import platform
import os
import json
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import transformers
from transformers import AdamW, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
#import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

import gc

import warnings
warnings.simplefilter('ignore')

pd.options.mode.chained_assignment = None

def set_seed(seed_value = 123):
	random.seed(seed_value)
	np.random.seed(seed_value)
	torch.manual_seed(seed_value)
	torch.cuda.manual_seed(seed_value)

# Remove Whitespace + Lowercase
def white_lower(x):
	for i in range(len(x.columns)):
		x[x.columns[i]] = x[x.columns[i]].str.replace(' ','').str.lower()
	return x

if torch.cuda.is_available():
    print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    DEVICE = torch.device('cuda:0')
else:
    print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
    DEVICE = torch.device('cpu')

label2id = {'ALIVE': 0, 'DEAD': 1}
id2label = {0: 'ALIVE', 1: 'DEAD'}

class Config:
	EPO = 10
	ML = 70
	BS = 16
	LR = 5e-5
	EPS = 1e-8
	SCALER = GradScaler()
	NS = 10

class DTSET(Dataset):
	def __init__(self, values, status, tokenizer, max_len):
			self.values = values
			self.status = status
			self.tokenizer = tokenizer
			self.max_len = max_len

	def __len__(self):
		return len(self.status)

	def __getitem__(self, idx):
		values = str(self.values[idx])
		status = self.status[idx]

		encode_values = self.tokenizer.encode_plus(
			values,
			truncation = True,
			add_special_tokens = True,
			max_length = self.max_len,
			pad_to_max_length = True,
			return_attention_mask = True,
			return_tensors = 'pt'
			)

		return {
			'ids': encode_values['input_ids'].flatten(),
			'mask': encode_values['attention_mask'].flatten(),
			'status': torch.tensor(status, dtype=torch.long)
		}

def create_dataloader(df, tokenizer, max_len, batch_size):
	ds = DTSET(
		values = df['values'].to_numpy(),
		status = df['status'].to_numpy(),
		tokenizer = tokenizer,
		max_len = max_len
	)

	return DataLoader(
		ds,
		batch_size = batch_size,
		num_workers = 0,
		shuffle = True,
		pin_memory = True
	)

def flat_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=1).flatten()
	labels_flat = labels.flatten()
	return np.sum(pred_flat == labels_flat) / len(labels_flat)

def yield_optimizer(model):
	optimizer = AdamW(model.parameters(), lr = Config.LR, eps = Config.EPS)
	return optimizer

class Trainer:
	def __init__(self, model, optimizer, scheduler, train_dataloader, test_dataloader, device):
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.train_data = train_dataloader
		self.test_data = test_dataloader
		self.scaler = Config.SCALER
		self.device = device
		
	def train_epoch(self):
		prog_bar = tqdm(enumerate(self.train_data), total = len(self.train_data))
		self.model.train()
		total_train_loss = 0

		for step, batch in prog_bar:
			b_ids = batch['ids'].to(self.device)
			b_mask = batch['mask'].to(self.device)
			b_status = batch['status'].to(self.device)
			
			self.optimizer.zero_grad()
		
			loss, logits = self.model(b_ids, token_type_ids = None, attention_mask = b_mask, labels = b_status, return_dict = False)

			total_train_loss += loss.item()

			logits = logits.detach().cpu().numpy()
						
			self.scaler.scale(loss).backward()
			self.scaler.step(self.optimizer)
			self.scaler.update()
			self.scheduler.step()

		avg_train_loss = total_train_loss / len(self.train_data)
		return avg_train_loss

	def valid_epoch(self):
		prog_bar = tqdm(enumerate(self.test_data), total = len(self.test_data))
		self.model.eval()
		total_val_loss = 0
		total_val_acc = 0
		total_kappa_score = 0

		with torch.no_grad():
			for step, batch in prog_bar:
				b_ids = batch['ids'].to(self.device)
				b_mask = batch['mask'].to(self.device)
				b_status = batch['status'].to(self.device)

				(loss, logits) = self.model(b_ids, token_type_ids = None, attention_mask = b_mask, labels = b_status, return_dict = False)

				total_val_loss += loss.item()

				logits = logits.detach().cpu().numpy()
				act_status = b_status.to('cpu').numpy()

				# calculate cohen-kappa
				if len(set(np.argmax(logits, axis=1).flatten() + act_status.flatten())) == 1:
					total_kappa_score += 1
				else:
					total_kappa_score += cohen_kappa_score(np.argmax(logits, axis=1).flatten(), act_status.flatten())

				total_val_acc += flat_accuracy(logits, act_status)

			avg_val_accuracy = total_val_acc / len(self.test_data)
			avg_val_loss = total_val_loss / len(self.test_data)
			avg_kappa_score = total_kappa_score / len(self.test_data)

		return avg_val_accuracy, avg_val_loss, avg_kappa_score

	def get_model(self):
		return self.model

if __name__ == '__main__':
	set_seed(123)
	if torch.cuda.is_available():
		print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
		DEVICE = torch.device('cuda:0')
	else:
		print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
		DEVICE = torch.device('cpu')

	# DATA PREPROCESSING
	dtrain = pd.read_csv('data/train.csv')

	X_train = dtrain.loc[:, dtrain.columns != 'V26']
	y_train = dtrain.loc[:, dtrain.columns == 'V26']

	# Convert all data to string
	X_train = X_train.astype(str)
	y_train = y_train.astype(str)

	# Lowercase + remove Whitespace
	X_train = white_lower(X_train)
	y_train = white_lower(y_train)

	# Mapping
	X_train['values'] = X_train['V2'].map(str) + ' ' + X_train['V3'].map(str) + ' ' + X_train['V4'].map(str) + ' ' + X_train['V5'].map(str) + ' ' + X_train['V6'].map(str) + ' ' + X_train['V7'].map(str) + ' ' + X_train['V8'].map(str) + ' ' + X_train['V9'].map(str) + ' ' + X_train['V10'].map(str) + ' ' + X_train['V11'].map(str) + ' ' + X_train['V12'].map(str) + ' ' + X_train['V13'].map(str) + ' ' + X_train['V14'].map(str) + ' ' + X_train['V15'].map(str) + ' ' + X_train['V16'].map(str) + ' ' + X_train['V17'].map(str) + ' ' + X_train['V18'].map(str) + ' ' + X_train['V19'].map(str) + ' ' + X_train['V20'].map(str) + ' ' + X_train['V21'].map(str) + ' ' + X_train['V22'].map(str) + ' ' + X_train['V23'].map(str) + ' ' + X_train['V24'].map(str) + ' ' + X_train['V25'].map(str)
	train = X_train.join(y_train)

	# Subset
	train = train[['V26','values']]
	train = pd.DataFrame(train)
	train = train.rename(columns={'V26':'status'})
	train['status'] = train['status'].replace(['alive','dead'],['0','1'])
	train['status'] = train['status'].astype('int64')

	model_list = ['bert-base-uncased', 'roberta-base', 'albert-base-v2', 'bert-base-uncased']
	model_name = ['bert', 'roberta', 'albert', 'bert+']

	#c = 0

	#PLOTTER (LOSS)
	def plotter(history, fold, mpath, mname, testing = False):
		if testing == True:
			labels = ['TRAINING', 'TESTING']
			fig_loss, axis_loss = plt.subplots(2, 5)
			fig_loss.set_size_inches(18.5, 10.5)
			for f in range(fold):
				if f < 5:
					axis_loss[0, f].plot(history[f'train_loss_f{f}'])
					axis_loss[0, f].plot(history[f'val_loss_f{f}'])
					axis_loss[0, f].set_title(f'FOLD {f + 1}')
					axis_loss[0, f].set(xlabel='Epoch', ylabel='Loss')
				else:
					axis_loss[1, f - 5].plot(history[f'train_loss_f{f}'])
					axis_loss[1, f - 5].plot(history[f'val_loss_f{f}'])
					axis_loss[1, f - 5].set_title(f'FOLD {f + 1}')
					axis_loss[1, f - 5].set(xlabel='Epoch', ylabel='Loss')
			fig_loss.subplots_adjust(wspace = 0.4, hspace = 0.3)
			fig_loss.legend(axis_loss, labels = labels, loc = "lower center")
			#fig_loss.show()
			fig_loss.savefig(f'{mpath}/{mname}_transformer_traintest_loss', dpi = 320)
		else:
			labels = ['TRAINING', 'VALIDATION']
			fig_loss, axis_loss = plt.subplots(2, 5)
			fig_loss.set_size_inches(18.5, 10.5)
			for f in range(fold):
				if f < 5:
					axis_loss[0, f].plot(history[f'train_loss_f{f}'])
					axis_loss[0, f].plot(history[f'val_loss_f{f}'])
					axis_loss[0, f].set_title(f'FOLD {f + 1}')
					axis_loss[0, f].set(xlabel='Epoch', ylabel='Loss')
				else:
					axis_loss[1, f - 5].plot(history[f'train_loss_f{f}'])
					axis_loss[1, f - 5].plot(history[f'val_loss_f{f}'])
					axis_loss[1, f - 5].set_title(f'FOLD {f + 1}')
					axis_loss[1, f - 5].set(xlabel='Epoch', ylabel='Loss')
			fig_loss.subplots_adjust(wspace = 0.4, hspace = 0.3)
			fig_loss.legend(axis_loss, labels = labels, loc = "lower center")
			#fig_loss.show()
			fig_loss.savefig(f'{mpath}/{mname}_transformer_training_loss', dpi = 320)

	#set_seed(123)
	f_loss_check = []

	kf = StratifiedKFold(n_splits = Config.NS, shuffle = True)
	n_bins = int(np.floor(1 + np.log2(len(train))))
	train.loc[:, 'bins'] = pd.cut(train['status'], bins = n_bins, labels = False)

	for c in range(4):
		torch.cuda.set_per_process_memory_fraction(0.5, 0)
		torch.cuda.empty_cache()

		avg_train_loss = []
		avg_kappa_scores = []
		avg_val_loss = []
		avg_val_acc = []
		history  = defaultdict(list)
		
		mod_name = model_name[c]
		mod = model_list[c]
		
		config = AutoConfig.from_pretrained(mod, label2id = label2id, id2label = id2label)
		model = AutoModelForSequenceClassification.from_pretrained(mod, config = config)
		tokenizer = AutoTokenizer.from_pretrained(mod)

		# getting uniques + adding to vocab
		if c == 3:
			v = list(pd.unique(X_train[X_train.columns[1:23]].values.ravel('K')))
			new_vocab = set(v) - set(tokenizer.vocab.keys())
			tokenizer.add_tokens(list(new_vocab))
			model.resize_token_embeddings(len(tokenizer))

		for fold, (train_idx, test_idx) in enumerate(kf.split(X = train, y = train['bins'].values)):
			print(f"\nFold no. {fold}")
			print(f"{'-'*30}\n")

			train_data = train.iloc[train_idx]
			test_data = train.iloc[test_idx]

			trn = create_dataloader(train_data, tokenizer, Config.ML, Config.BS)
			tst = create_dataloader(test_data, tokenizer, Config.ML, Config.BS)

			model.to(DEVICE)

			nb_steps = int(len(train_data) / Config.BS * Config.EPO)
			optimizer = yield_optimizer(model)
			scheduler = get_linear_schedule_with_warmup(
				optimizer,
				num_warmup_steps = 0,
				num_training_steps = nb_steps
			)

			trainer = Trainer(
				model = model,
				optimizer = optimizer,
				scheduler = scheduler,
				train_dataloader = trn,
				test_dataloader = tst,
				device = DEVICE)

			best_loss = 100
			training_stats = []

			rec_train_loss = []
			rec_kappa_score = []
			rec_val_loss = []
			rec_val_acc = []

			for epoch in range(1, Config.EPO + 1):
				print(f"\n{'--'*5} EPOCH: {epoch} {'--'*5}\n")
				train_loss = trainer.train_epoch()
				print(f"Train_Loss {train_loss:.2f}")

				val_acc, val_loss, kappa_score = trainer.valid_epoch()
				print(f"Val_Acc {val_acc:.2f} || Val_Loss {val_loss:.2f} || Kappa_Score {kappa_score:.2f}")
				
				# <need to record> #
				training_stats.append(
					{	
						'fold' : fold,
						'epoch' : epoch,
						'training_loss' : round(train_loss, 4),
						'kappa_score' : round(kappa_score, 4),
						'validation_loss' : round(val_loss, 4),
						'validation_accuracy' : round(val_acc, 4)
					}
				)

				rec_train_loss.append(train_loss)
				rec_kappa_score.append(kappa_score)
				rec_val_loss.append(val_loss)
				rec_val_acc.append(val_acc)

				history[f'train_loss_f{fold}'].append(train_loss)
				history[f'train_kappa_score_f{fold}'].append(kappa_score)
				history[f'val_loss_f{fold}'].append(val_loss)
				history[f'val_acc_f{fold}'].append(val_acc)

				if val_loss < best_loss:
					best_loss = val_loss
					print(f"Recording model with current best loss of: {val_loss:.2f}")
					best_model = model

			print(f"Best LOSS in fold: {fold} was: {best_loss:.2f}")

			avg_train_loss.append(sum(rec_train_loss)/len(rec_train_loss))
			avg_kappa_scores.append(sum(rec_kappa_score)/len(rec_kappa_score))
			avg_val_loss.append(sum(rec_val_loss)/len(rec_val_loss))
			avg_val_acc.append(sum(rec_val_acc)/len(rec_val_acc))

			mod_path = f'automodel/{mod_name}' 
			if not os.path.exists(mod_path):
				os.makedirs(mod_path)
				print("Creating directory...")

			scores_avg_train_loss = f'{mod_path}/avg_train_loss.txt'
			scores_avg_kappa_scores = f'{mod_path}/avg_kappa_scores.txt'
			scores_avg_val_loss = f'{mod_path}/avg_val_loss.txt'
			scores_avg_val_acc = f'{mod_path}/avg_val_acc.txt'
			print("Saving scores...")

			if os.path.isfile(scores_avg_train_loss) and os.access(scores_avg_train_loss, os.R_OK):
				print("File exists and will be overwritten...")
				os.remove(scores_avg_train_loss)
				with open(scores_avg_train_loss, mode = 'w') as f:
					for i in avg_train_loss:
						f.write("%s\n" % i)
			else:
				print("File does not exist and will be created...")
				with open(scores_avg_train_loss, mode = 'w') as f:
					for i in avg_train_loss:
						f.write("%s\n" % i)

			if os.path.isfile(scores_avg_kappa_scores) and os.access(scores_avg_kappa_scores, os.R_OK):
				print("File exists and will be overwritten...")
				os.remove(scores_avg_kappa_scores)
				with open(scores_avg_kappa_scores, mode = 'w') as f:
					for i in avg_kappa_scores:
						f.write("%s\n" % i)
			else:
				print("File does not exist and will be created...")
				with open(scores_avg_kappa_scores, mode = 'w') as f:
					for i in avg_kappa_scores:
						f.write("%s\n" % i)

			if os.path.isfile(scores_avg_val_loss) and os.access(scores_avg_val_loss, os.R_OK):
				print("File exists and will be overwritten...")
				os.remove(scores_avg_val_loss)
				with open(scores_avg_val_loss, mode = 'w') as f:
					for i in avg_val_loss:
						f.write("%s\n" % i)
			else:
				print("File does not exist and will be created...")
				with open(scores_avg_val_loss, mode = 'w') as f:
					for i in avg_val_loss:
						f.write("%s\n" % i)

			if os.path.isfile(scores_avg_val_acc) and os.access(scores_avg_val_acc, os.R_OK):
				print("File exists and will be overwritten...")
				os.remove(scores_avg_val_acc)
				with open(scores_avg_val_acc, mode = 'w') as f:
					for i in avg_val_acc:
						f.write("%s\n" % i)
			else:
				print("File does not exist and will be created...")
				with open(scores_avg_val_acc, mode = 'w') as f:
					for i in avg_val_acc:
						f.write("%s\n" % i)

			output_dir = f'{mod_path}/save_fold_{fold}'
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)
			print("Saving best model...")
			saving_model = best_model.module if hasattr(best_model, 'module') else best_model
			saving_model.save_pretrained(output_dir)
			tokenizer.save_pretrained(output_dir)

			stats_dir =  f'{output_dir}/outputs'
			stats_save = f'{stats_dir}/training_stats_fold_{fold}.txt'
			if not os.path.exists(stats_dir):
				os.makedirs(stats_dir)
			if os.path.isfile(stats_save) and os.access(stats_save, os.R_OK):
				print("File exists and will be overwritten...")
				os.remove(stats_save)
				with open(stats_save, mode = 'w') as f:
					f.write(str(training_stats))
			else:
				print("Creating file...")
				print("Saving outputs...")
				with open(stats_save, mode = 'w') as f:
					f.write(str(training_stats))

			# Garbage Collection
			del trainer
			del train_data
			del test_data
			del trn
			del tst
			del nb_steps
			del optimizer
			del scheduler
			del best_loss
			del train_loss
			del val_loss
			del val_acc
			del training_stats
			del rec_train_loss
			del rec_kappa_score
			del rec_val_loss
			del rec_val_acc
			del best_model
			del output_dir
			del stats_dir
			del stats_save
			gc.collect()

		plotter(history, 10, mod_path, mod_name)

		del model
		del tokenizer
		gc.collect()
			
		# Retrieve fold with lowest average loss
		fold = avg_val_loss.index(min(avg_val_loss))
		print(avg_val_loss.index(min(avg_val_loss)))
		f_loss_check.append(fold)

	#TESTING
	set_seed(123)
	plt.clf()
	dvalid = pd.read_csv('data/valid.csv')
	X_val = dvalid.loc[:, dvalid.columns != 'V26']
	y_val = dvalid.loc[:, dvalid.columns == 'V26']
	X_val = X_val.astype(str)
	y_val = y_val.astype(str)
	X_val = white_lower(X_val)
	y_val = white_lower(y_val)
	X_val['values']= X_val['V2'].map(str) + ' ' + X_val['V3'].map(str) + ' ' + X_val['V4'].map(str) + ' ' + X_val['V5'].map(str) + ' ' + X_val['V6'].map(str) + ' ' + X_val['V7'].map(str) + ' ' + X_val['V8'].map(str) + ' ' + X_val['V9'].map(str) + ' ' + X_val['V10'].map(str) + ' ' + X_val['V11'].map(str) + ' ' + X_val['V12'].map(str) + ' ' + X_val['V13'].map(str) + ' ' + X_val['V14'].map(str) + ' ' + X_val['V15'].map(str) + ' ' + X_val['V16'].map(str) + ' ' + X_val['V17'].map(str) + ' ' + X_val['V18'].map(str) + ' ' + X_val['V19'].map(str) + ' ' + X_val['V20'].map(str) + ' ' + X_val['V21'].map(str) + ' ' + X_val['V22'].map(str) + ' ' + X_val['V23'].map(str) + ' ' + X_val['V24'].map(str) + ' ' + X_val['V25'].map(str)
	val = X_val.join(y_val)

	val = val[['V26','values']]
	val = pd.DataFrame(val)
	val = val.rename(columns = {'V26':'status'})
	val['status'] = val['status'].replace(['alive','dead'],['0','1'])
	val['status'] = val['status'].astype('int64')

	def BC_Predict(model, val_data, dev):
		prog_bar = tqdm(enumerate(val_data), total = len(val_data))
		prd = []
		act = []

		with torch.no_grad():
			for step, batch in prog_bar:
				b_ids = batch['ids'].to(dev)
				b_mask = batch['mask'].to(dev)
				b_status = batch['status'].to(dev)

				(_, logits) = model(b_ids, token_type_ids = None, attention_mask = b_mask, labels = b_status).to_tuple()

				logits = logits.detach().cpu().numpy()
				act_status = b_status.to('cpu').numpy()

				logits_flat = np.argmax(logits, axis=1).flatten()
				act_flat = act_status.flatten()

				prd.extend(logits_flat)
				act.extend(act_flat)
		return prd, act

	def print_metrics(pred, actual, mod_name, mod_path):
		new_dict = {}
		cm = confusion_matrix(actual, pred, labels = [0, 1])
		print("CONFUSION MATRIX")
		print(cm)
		cr = classification_report(actual, pred, labels = [0, 1], digits = 4, output_dict = True)
		#df = pd.DataFrame(cr).transpose()
		#df.to_csv(f'{mod_path}/classification_report.csv', index = True)
		#print("CLASSIFICATION REPORT")
		print(cr)

		fpr, tpr, thresholds = roc_curve(actual, pred)
		auc = round(roc_auc_score(actual, pred), 4)
		#print("AUC")
		#print(auc)
		plt.plot(fpr, tpr, label = f'{mod_name}_AUC = %0.2f' % auc)
		#plt.xlabel('1 - Specificity')
		#plt.ylabel('Sensitivity')
		#plt.show()
		#plt.savefig(f'{path}/model_roc', dpi = 320)

		total=sum(sum(cm))
		accuracy=round((cm[0,0]+cm[1,1])/total, 4)
		print ('Accuracy : ', accuracy)
		new_dict['Accuracy'] = accuracy

		"sensitivity = round(cm[0,0]/(cm[0,0]+cm[0,1]), 4)"
		"print('Sensitivity : ', sensitivity)"

		specificity = round(cm[1,1]/(cm[1,0]+cm[1,1]), 4)
		print('Specificity : ', specificity)
		new_dict['Specificity'] = specificity

		"prevalence = round((cm[0,0]+cm[1,0])/total, 4)"
		"print('Prevalence : ', prevalence)"

		precision = round(cm[0,0]/(cm[0,0]+cm[1,0]), 4)
		print('Precision : ', precision)
		new_dict['Precision'] = precision

		recall = round(cm[0,0]/(cm[0,0]+cm[0,1]), 4)
		print('Recall : ', recall)
		new_dict['Recall'] = recall
		
		f1score = round(2*((precision*recall)/(precision+recall)), 4)
		print('F1 Score : ', f1score)
		new_dict['F1 Score'] = f1score

		df = pd.DataFrame(new_dict, index = [0]).transpose()
		df.to_csv(f'{mod_path}/classification_report.csv', index = True)

		return cm

	set_seed(123)
	for c in range(4):
		torch.cuda.set_per_process_memory_fraction(0.5, 0)
		torch.cuda.empty_cache()
		
		mod_name = model_name[c]
		mod_path = f'automodel/{mod_name}'
		#fold = int(input("INPUT FOLD NO.    :")) *manual input*
		fold = f_loss_check[c]
		path = f'{mod_path}/save_fold_{fold}'
		tokenizer = AutoTokenizer.from_pretrained(path, local_files_only = True)
		model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only = True)
		model.to(DEVICE)
			
		train_dl = create_dataloader(train, tokenizer, Config.ML, Config.BS)
		val_dl = create_dataloader(val, tokenizer, Config.ML, Config.BS)

		print(f'{mod_name}	Fold = {fold}')
		pred_val, actual_val = BC_Predict(model, val_dl, DEVICE)
		print_metrics(pred_val, actual_val, mod_name, mod_path)
		print('')

	plt.title('AUC-ROC CURVE')
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1], 'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('Sensitivity')
	plt.xlabel('1 - Specificity')
	plt.show()
	