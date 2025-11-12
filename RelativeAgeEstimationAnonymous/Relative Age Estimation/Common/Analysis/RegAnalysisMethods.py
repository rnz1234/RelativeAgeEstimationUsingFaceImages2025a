import torch
import numpy as np
import json
from tqdm import tqdm

def reg_pipeline_mae_and_hist_analysis(pipeline, data_loader, device, dataset_size, dataset_type, pipeline_is_model):
	print('running on validation set...')

	running_mae = 0.0
	running_mae_out_of_radius = 0.0
	running_mae_tot = 0.0
	running_madev = 0.0
	ignored = 0

	err_vec = []

	im2age_map = dict()


	for batch in tqdm(data_loader):
		inputs = batch['image'].to(device)
		ages = batch['age'].to(device).float()
		idxs = batch['idx'].to(device).float()

		with torch.no_grad():
			if pipeline_is_model:
				age_pred = pipeline(inputs)
			else:
				age_pred = pipeline.predict(inputs)#, bypass_diff=True, check_ranges=False)

		errs = age_pred - ages

		# import pdb
		# pdb.set_trace()
		for j in range(len(idxs)):
			im2age_map[int(idxs[j].cpu())] = float(age_pred[j].cpu())

		# import pdb
		# pdb.set_trace()

		err_vec += errs.cpu().numpy().tolist()
			
		running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)

		#if i % 100 == 50:
		# if i % 5 == 0:
		# 	print("progress : {:.2f}%".format((64*(i + 1) / dataset_size) * 100))

		# if i == 10:
		# 	break

		

	err_hist = np.histogram(np.array(err_vec), bins=100, density=True)

	# import pdb
	# pdb.set_trace()
	im2age_map_js = json.dumps(im2age_map)

	with open(f'im2age_map_{dataset_type}.json', 'w') as fmap:
		fmap.write(im2age_map_js)
		
	#plt.hist(np.array(err_vec), 50)


	total_mae_tot = running_mae / dataset_size

	print('Total MAE : {:.4f}'.format(total_mae_tot))
	print('Total STD DEV : {:.4f}'.format(np.std(np.array(err_vec))))

	# plt.plot(err_hist)
	# plt.show()

	return total_mae_tot


import json
import numpy as np
import matplotlib.pyplot as plt


def age_predict_stats(dataset_metadata, dataset_indexes, im2age_map_batst):
	err_vec = []
	for i in range(len(dataset_metadata)):
		age_real = int(json.loads(dataset_metadata[i])['age'])
		age_pred = im2age_map_batst[str(dataset_indexes[i])]
		err_vec.append(age_real-age_pred)

	err_vec = np.array(err_vec)

	# import pdb
	# pdb.set_trace()

	print("----------------------------------------------")
	print("-- Age Predict (Original) - Stats")
	print("--")
	print("- MAE (mean abs error) : {}".format(np.mean(np.abs(err_vec))))
	print("- ME (mean error)      : {}".format(np.mean(err_vec)))
	print("- STDev (standard dev) : {}".format(np.std(err_vec)))

	plt.hist(np.array(err_vec), bins=20)
	plt.show()


def create_age_predict_preds_db(dataset_metadata, dataset_indexes, im2age_map_batst):
	err_vec = []
	for i in range(len(dataset_metadata)):
		age_real = int(json.loads(dataset_metadata[i])['age'])
		age_pred = im2age_map_batst[str(dataset_indexes[i])]
		err_vec.append(age_real-age_pred)

	err_vec = np.array(err_vec)

	# import pdb
	# pdb.set_trace()

	print("----------------------------------------------")
	print("-- Age Predict (Original) - Stats")
	print("--")
	print("- MAE (mean abs error) : {}".format(np.mean(np.abs(err_vec))))
	print("- ME (mean error)      : {}".format(np.mean(err_vec)))
	print("- STDev (standard dev) : {}".format(np.std(err_vec)))

	plt.hist(np.array(err_vec), bins=20)
	plt.show()


