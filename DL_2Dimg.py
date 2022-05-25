# Function to train and evaluate 2D classification and regression DL models
# Márton Kolossváry, MD PhD - (c) 2022

import os
import pandas as pd
import torch  as th
from   fastai.vision.all import *
from   PIL import Image
from   torchvision import transforms


# Initialize =====
spec = pd.read_csv(sys.argv[1])
n_models = spec.shape[1]-2


# Cycle through all model specifications =====
for i_model in range(1, n_models+1):

  ## Get and set default attributes -----
  att_dic = dict(zip(spec.iloc[:, 0].to_list(), spec.iloc[:, i_model+1].to_list()))
  print("Starting analyses for: " + att_dic['model_f_name'])
  f_db = pd.read_csv(att_dic['path'] + att_dic['csv_fname'])
  defaults.cpus = int(att_dic['num_workers'])
  th.cuda.set_device(int(att_dic['device']))

  ## Initialize transformations -----
  item_tfms = Resize(size = tuple(map(int, att_dic['size'].split(', '))),
    method = att_dic['method'], pad_mode = att_dic['pad_mode'])

  batch_tfms = aug_transforms(do_flip = bool(att_dic['do_flip']), flip_vert = bool(att_dic['flip_vert']),
    max_rotate = float(att_dic['max_rotate']), min_zoom = float(att_dic['min_zoom']), max_zoom = float(att_dic['max_zoom']),
    max_warp = float(att_dic['max_warp']), p_affine = float(att_dic['p_affine']),
    max_lighting = float(att_dic['max_lighting']), p_lighting = float(att_dic['p_lighting']), pad_mode = att_dic['pad_mode'],
    align_corners=True, batch=False, min_scale=1.0, mult=1.0, xtra_tfms=None, size=None, mode='bilinear')

  ## Initialize outcomes -----
  if att_dic['outcome'] == "unilabel":
    y_block = CategoryBlock
    label_delim = None
    f_db[att_dic['label_col']] = f_db[att_dic['label_col']].astype('str')
  elif att_dic['outcome'] == "multilabel":
    y_block = MultiCategoryBlock
    label_delim = str(att_dic['label_delim'])
    f_db[att_dic['label_col']] = f_db[att_dic['label_col']].astype('str')
  else:
    y_block = RegressionBlock
    label_delim = None
    f_db[att_dic['label_col']] = f_db[att_dic['label_col']].astype('float')

  ## Remove test cases if training or finetuning -----
  if bool(int(att_dic['train'])):

    ### Create model directory if not present
    if not os.path.isdir(att_dic['path'] + att_dic['model_dir'] + "/"):
      os.mkdir(att_dic['path'] + att_dic['model_dir'] + "/")

    ### Remove rows that don't have data in valid_col (as those are test cases)
    print("Removing rows without data in the label column: " + att_dic['label_col'])
    f_db = f_db.dropna(subset = [att_dic['valid_col']]).reset_index(drop=True)
    f_db[att_dic['valid_col']] = f_db[att_dic['valid_col']].astype('boolean')

    ### Check if all files exist that are needed for training and validation
    for imgs in f_db.loc[:, att_dic['fn_col']]:
      if(not os.path.exists("/" + imgs)):
        print("Terminating script! Image does not exist: \n" + imgs)
        quit()
    print("All files exist for training and tuning!")

  ## Initialize files loading -----
  dls = ImageDataLoaders.from_df(df = f_db, bs = int(att_dic['bs']), num_workers = int(att_dic['num_workers']),
    fn_col = att_dic['fn_col'], label_col = att_dic['label_col'], valid_col = att_dic['valid_col'],
    item_tfms = item_tfms, batch_tfms = batch_tfms, y_block = y_block, label_delim = label_delim,
    seed=42, valid_pct=0, shuffle=True, folder=None, path='/', suff='')

  ## Initialize learner -----
  if bool(int(att_dic['load_existing'])):
    learn = cnn_learner(dls = dls, arch = eval(att_dic['arch']), path = att_dic['path'], model_dir = att_dic['model_dir']).load(
          file = att_dic['path'] + att_dic['model_dir'] + "/" + att_dic['model_f_name'])
    if bool(int(att_dic['train'])):
      att_dic['model_f_name'] = att_dic['model_f_name'] + "_finetune"
  else:
    learn = cnn_learner(dls, arch = eval(att_dic['arch']), path = att_dic['path'], model_dir = att_dic['model_dir'],
      loss_func = eval(att_dic['loss_func']), opt_func = eval(att_dic['opt_func']), moms = tuple(map(float, att_dic['moms'].split(', '))),
      pretrained=True, normalize=True, metrics=None, n_out=None, config=None, lr=0.001, wd=None, wd_bn_bias=False, train_bn=True,
      splitter=None, cut=None, n_in=3, init=nn.init.kaiming_normal_,
      custom_head=None, concat_pool=True, lin_ftrs=None, ps=0.5, first_bn=True, bn_final=False, lin_first=False, y_range=None, cbs=None)

  ## Initialize callbacks -----
  cbss = ["CSVLogger(fname = att_dic['path'] + att_dic['model_dir'] + '/' + att_dic['model_f_name'] + '_log.csv', append = True)",
          "SaveModelCallback(monitor = 'valid_loss', fname = att_dic['model_f_name'], with_opt = True, reset_on_fit = False)"]
  cbss = cbss + [val for key,val in att_dic.items() if key in ["cbs_" + str(x) for x in range(100)]]
  cbss = [x for x in cbss if str(x) != 'nan']
  for cbs in cbss:
    learn.add_cb(eval(cbs))

  ## TRAINING =====
  if bool(int(att_dic['train'])):
    print("Will train " + att_dic['model_f_name'] + " model using " + str(f_db.shape[0] - sum(f_db.loc[:, att_dic['valid_col']])) +
    " training samples and " + str(sum(f_db.loc[:, att_dic['valid_col']])) + " tuning samples")

    ### Freeze layers
    learn.freeze_to(n = int(att_dic['freeze_to']))

    ### Find optimal learning rates
    suggest_funcs       = tuple(map(eval, att_dic['suggest_funcs'].split(', ')))
    suggest_funcs_names = tuple(map(str, att_dic['suggest_funcs'].split(', ')))
    print("Finding optimal learning rate using " + ", ".join(map(str, suggest_funcs_names)))
    lrs = learn.lr_find(suggest_funcs = suggest_funcs, start_lr=1e-07, end_lr=10, num_it=100, stop_div=True, show_plot=False)
  
    ### Fit using different strategies
    if att_dic['lr_type'] == 'discriminative':
      print("Fitting using discriminative learning using min-max range of optimal learning rates using " + att_dic['n_epoch'] + " epochs")
      learn.fit_one_cycle(n_epoch = int(att_dic['n_epoch']), lr_max = slice(min(lrs), max(lrs))) #Discriminative learning between slice start and stop
    else:
      for lr in range(0, len(lrs)):
        print("Fitting using " + str(lr+1) + "/" + str(len(lrs)) + " different learning rates using " + att_dic['n_epoch'] + " epochs each")
        if lr > 0: #Load previous sequential learning result if more the one cycles are present
          learn.load(att_dic['path'] + att_dic['model_dir'] + "/" + att_dic['model_f_name'])
          lrs = learn.lr_find(suggest_funcs = suggest_funcs, start_lr=1e-07, end_lr=10, num_it=100, stop_div=True, show_plot=False) #Re-evaluate LR
        learn.fit_one_cycle(n_epoch = int(att_dic['n_epoch']), lr_max = lrs[lr],
          div=25.0, div_final=100000.0, pct_start=0.25, wd=None, moms=None, cbs=None, reset_opt=False)

  ## Write log of model -----
  file = open(att_dic['path'] + att_dic['model_dir'] + "/" + att_dic['model_f_name'] + "_summary.txt", "w")
  file.write(learn.summary())
  file.write("\n\n\n\n\n")
  file.write(str(learn.model))
  file.close

  ## If trained and predict or GradCAM than reload all images including test cases ----
  if bool(int(att_dic['train']) and (bool(int(att_dic['predict'])) or bool(int(att_dic['grad_cam'])))):
    f_db = pd.read_csv(att_dic['path'] + att_dic['csv_fname'])

    for imgs in f_db.loc[:, att_dic['fn_col']]:
      if(not os.path.exists("/" + imgs)):
        print("Terminating script! Image does not exist: \n" + imgs)
        quit()
    print("All files exist for predictions and/or GradCAMs!")

  ## PREDICTIONS =====
  if bool(int(att_dic['predict'])):
    print("Creating predictions for total dataset")

    ### Create database to get predictions
    dl = learn.dls.test_dl("/" + f_db.loc[:, att_dic['fn_col']], rm_type_tfms=None, with_labels=False)
    if bool(int(att_dic['tta'])):
      preds, _ = learn.tta(dl = dl, n = int(att_dic['tta_n']), beta = float(att_dic['tta_beta']), use_max = bool(int(att_dic['use_max'])),
        item_tfms=None, batch_tfms=None)
    else: 
      preds, _ = learn.get_preds(dl = dl,
        with_input=False, with_decoded=False, with_loss=False, act=None, inner=False, reorder=True, cbs=None, save_preds=None, save_targs=None, concat_dim=0)

    ### Concatenate with database
    db = f_db.loc[:, [att_dic['fn_col'], att_dic['label_col']]]
    preds = pd.DataFrame((preds.numpy()))
    preds = preds.add_prefix("pred_")
    out = pd.concat([db.reset_index(drop=True), preds], axis=1)
    out.to_csv(att_dic['path'] + att_dic['model_dir'] + "/" + att_dic['model_f_name'] + "_pred.csv", index=False)
    print("Predictions done for model: " + att_dic['model_f_name'])

  ## GradCAM =====
  if bool(int(att_dic['grad_cam'])):
    print("Creating GradCAM images on specified images")

    ### Initialize parameters for GradCAM
    cmap  = att_dic['cmap']
    alpha = float(att_dic['alpha'])
    model_layer = att_dic['model_layer']

    ### Create GradCAM directory if not present
    if not os.path.isdir(att_dic['path'] + att_dic['gradcam_dir'] + "/"):
      os.mkdir(att_dic['path'] + att_dic['gradcam_dir'] + "/")

    ### Function to register hook on given layer
    def hooked_backward(x, y, model, model_layer):
      xb= x[None, :]
      with hook_output(eval("model" + model_layer)) as hook_a:
          with hook_output(eval("model" + model_layer), grad=True) as hook_g:
              preds = model(xb)
              preds[0,int(y)].backward()
      return hook_a,hook_g

    ### Create variables for which GradCAMS are created
    f_db = f_db[f_db[att_dic['grad_cam_col']] == 1].reset_index(drop=True)
    dl = learn.dls.test_dl("/" + f_db.loc[:, att_dic['fn_col']], rm_type_tfms=None, with_labels=False, bs = f_db.shape[0])
    model = learn.model.eval()
    x = dl.one_batch()

    for i in range(0, f_db.shape[0]):
      #### Calculate activations and gradients
      if att_dic['gradcam_outcome'] == "truth":
        y = int(f_db.loc[i, att_dic['label_col']])
      else:
        y = int(att_dic['gradcam_outcome'])

      hook_a,hook_g = hooked_backward(x[0][i], y, model, model_layer) # Last layer before average pooling
      acts  = hook_a.stored[0].cpu()
      grads = hook_g.stored[0][0].cpu()

      #### Create activation and gradient maps according to summary function
      if att_dic['sum_func'] == "mean":
          acts_summ  = acts.mean(0)
          grads_summ = grads.mean(1)
      elif att_dic['sum_func'] == "median":
          acts_summ = acts.median(0).values
          grads_summ = grads.median(1).values
      elif att_dic['sum_func'] == "max":
          acts_summ = acts.max(0).values
          grads_summ = grads.max(1).values
      elif att_dic['sum_func'] == "mean_max":
          acts_summ = (acts.mean(0) + acts.max(0).values)/2
          grads_summ = (grads.mean(1) + grads.max(1).values)/2
      acts_x_grads = acts*grads_summ[..., None]
      acts_x_grads = acts_x_grads.mean(0)

      #### Normalize
      acts_summ = (acts_summ - acts_summ.min())/(acts_summ.max() - acts_summ.min())
      grads_summ = (grads_summ - grads_summ.min())/(grads_summ.max() - grads_summ.min())
      acts_x_grads = (acts_x_grads - acts_x_grads.min())/(acts_x_grads.max() - acts_x_grads.min())

      #### Select given type of heatmap
      if   att_dic['heatmap'] == "activation":
          heatmap = acts_summ
      elif att_dic['heatmap'] == "gradient":
          heatmap = grads_summ
      elif att_dic['heatmap'] == "activation_x_gradient":
          heatmap = acts_x_grads

      #### Open original image
      img = Image.open("/" + f_db.loc[i, att_dic['fn_col']])
      img = img.convert('RGB')

      #### Scale up heatmap
      heatmap = np.uint8(heatmap*255)
      heatmap = transforms.ToPILImage()(heatmap).convert('L')
      heatmap = heatmap.resize((img.shape[1], img.shape[0]), resample = Image.BILINEAR)

      #### Convert to RGB
      cm = plt.get_cmap(cmap)
      im = np.array(heatmap)
      im = cm(im)
      im = np.uint8(im*255)
      im = Image.fromarray(im).convert('RGB')

      #### Blend and save
      new_img = Image.blend(img, im, alpha)
      new_img_name = f_db.loc[i, att_dic['fn_col']]
      new_img_name = new_img_name[new_img_name.rindex("/")+1:]
      new_img_name = att_dic['path'] + att_dic['gradcam_dir'] + "/" + new_img_name[:-4] + "_GradCAM_" + att_dic['model_f_name'] + ".png"
      new_img.save(new_img_name)

      print("Done with " + str(i+1) +  "/" + str(f_db.shape[0]) + ": " + new_img_name)