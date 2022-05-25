<h1 align="left"> 2D Image classification and regression using fastai </h1>

The current document describes the use of **DL_2Dimg.py** script. The script requires a *database.csv* containing the image file paths and outcomes (detailed below), and a *spec.csv* specification file which determines what kind of models should be trained/evaluated. Sample files are provided in this folder.

<h2 align="left"> :notebook: Implemented functionalities </h2>

- **Multiple models**: Using the specification csv, run any number and combination of models using different architectures, images, training splits etc.
- **Training and evaluation**: Models can be trained and also evaluated using the script by providing predictions and class activation maps.
- **Inputs**: any commonly used image format: '.cpt', '.rgb', '.djv', '.pat', '.pgm', '.cr2', '.ico', '.jp2', '.djvu', '.cdt', '.wbmp', '.psd', '.png', '.svg', '.tiff', '.xbm', '.orf', '.pnm', '.ief', '.crw', '.jpg2', '.pcx', '.jpe', '.bmp', '.jpm', '.ras', '.jpg', '.tif', '.gif', '.jpf', '.cdr', '.art', '.jng', '.nef', '.svgz', '.xwd', '.xpm', '.ppm', '.jpx', '.erf', '.pbm', '.jpeg'.
- **Architectures**: Any supported by fastai from: https://github.com/fastai/fastai/blob/master/fastai/vision/models/tvm.py.
- **Outcomes**: Unilabel, Multilabel and Continuous.
- **Augmentations**: Any used by *Resize* and *aug_transforms*.
- **Optimization**: Any implemented loss (https://docs.fast.ai/losses.html) and optimizer (https://docs.fast.ai/optimizer.html) function may be used from fastai.
- **Hyperparameters**: Set all hyperparameters using specification file.
- **Learning rate**: Finds optimal learning rates using *lr_find*.
- **Learning strategies**: Multiphase (*sequential*) and *discriminative* learning are currently supported.
- **Transfer learning**: Flexibly set which parameter groups to freeze.
- **Test-time augmentation**: Calculate predictions using TTA.
- **Callbacks**: Set as many callbacks as needed.
- **Hardware**: Freely set which GPU to use, and how many CPUs to use during all processes.
- **Logging**: Provides model summaries, epoch fit statistics and predictions in a standard way for all models.
- **CAMS**: Provides heatmaps for activations, gradients of gradient-weighted class activation (GradCAM) for the specified layer.

<h2 align="left"> :computer: How to use </h2>

The following packages need to be installed: os, pandas, pytorch, PIL, torchvision and fastai.

Modify the **spec.csv** file to run. A sample file is provided in the repository for different outcome types.

Run **DL_2Dimg.py** in the terminal providing the full file path to the specification.csv:

*Example: :~$ DL_2Dimg.py /mnt/md0/my_user/project/spec.csv*

The script can be used to train a model and to predict using the same instance and possibly provide GradCAM images for selected cases, or to load a pretrained model and to just predict outcomes and/or create GradCAM images using the model. The script generates the following outputs:

-  *model_f_name.pth*: Saved model with optimizer state. Only provided if training is done. If only predictions are done than this already exists and is loaded. In case of fine tuning "_finetuning" is appended to the name.
-  *model_f_name_log.csv*: CSV file containing the training and validation loses per epoch for each learning strategy. Only provided if training is done.
-  *model_f_name_summary.txt*: Summary of the model architecture.
-  *model_f_name_pred.csv*: CSV containing the full file paths of all files in the provided *database* csv defined by the *fn_col*. Furthermore, the outcome variables are provided from the *label_col* of the *database* csv and as many columns as outcomes, which are named *pred_ABC*, where ABC is one of all possible classes in the *label_col* of the *database* csv.

When prediction or GradCAMS are done (irrespective of whether training was done prior to running the function or just before), all images are evaluated which are present in the *fn_col* column of *database* CSV. Therefore, for the simples workflow, create one *database* CSV with all the training, tuning and test data. In the *valid_col* column of *database* CSV insert *0* for training data, *1* for tunning data, and leave test cases empty. This way training will be done on cases labeled as *0* in the *valid_col*, tuning on *1s* and all images will be evaluated.

<h2 align="left"> :page_facing_up: Arguments of the specification file </h2>

The script works by specifying any number of models to train and/or to evaluate using a *specification* csv. Any number of additional columns may be present in the *database* csv. The script can do three things 1) train, 2) predict, 3) create GradCAM images. Any combination of these is valid. However, it is always advised to provide a full specification file, even if many of the arguments are not used in that particular case (eg learning rate if only predictions are done). A list of which arguments are used for only model predictions and or GradCAMs are provided at the end of this chapter.

<h3 align="left"> Database file format </h3>
CSV document with at least three columns.

- *file name column*: Column name containing full file paths to the images. Be aware that it must not start with "/", as it is automatically added.
- *Outcome column*: Column name containing labels. In case *outcome* is discrete, all unique occurrences are used as outcomes by converting the values to strings. For multilabel learning the outcome classes need to be divided by a character defined in the specification file using *label_delim*. If *outcome* is continuous, then the column is converted to float.
- *Validation colum*: Only used if training is done, however it is also required if only predictions or GradCAMS are done, but it is not used. Only accepted values in this column are 0, 1 of left empty. The column is converted to a boolean before use.

<h3 align="left"> Specification file format </h3>
CSV document, where first column are the arguments or headers of argument groups (to help legibility). The second column are brief definitions of the arguments. All further columns define models which are trained and/or evaluated. The column names of these additional columns is arbitrary.

<h4 align="left"> Argument list </h4>

**TASK SPECIFICATION**
- load_existing: *[1/0]* Is the model a existing model saved in *model_dir* with name *model_f_name*, or a raw model supported by fastai.
- train: *[1/0]* Should the model specified be trained, if load_existing is true than the model is fine-tuned.
- predict: *[1/0]* Should predictions be returned.
- grad_cam: *[1/0]* Should GradCam images be created using the model for rows indicated in *grad_cam_col*.

**GENERAL PARAMETERS**
- outcome: *[str]* The type of outcome. Can be: *unilabel* / *multilabel* / *continuous*.
- label_delim: *[str]* In case us multilabel learning the character dividing the different classes within the cells of *label_col*.
- model_f_name.: *[str]* Unique string used to name the resulting files:
  -  *model_f_name.pth*: Saved model with optimizer state. Only provided if training is done. If only predictions are done than this already exists and is loaded.
  -  *model_f_name_log.csv*: CSV file containing the training and validation loses per epoch for each learning strategy. Only provided if training is done.
  -  *model_f_name_summary.txt*: Summary of the model architecture.
  -  *model_f_name_pred.csv*: CSV containing the full file paths of all files in the provided *database* csv defined by the *fn_col*. Furthermore, the outcome variables is provided from the *label_col* of the *database* csv and as many columns as outcomes, which are named *pred_ABC*, where ABC is one of all possible classes in the *label_col* of the *database* csv.
- device: *[int]* The GPU device number provided by NVIDIA to run the model on. See nvidia-smi for available GPUs.
- num_workers: *[int]* How many CPUs should be used for all processes.
- model_dir: *[str]* Folder name within *path* where model outputs are saved and or loaded. Created if not present.

**DATABASE PARAMETERS**
- path: *[str]* Full path to folder containing *database* csv and where *model_dir* resides (or is created). This can change from model to model.
- csv_fname: *[str]* Filename of *database* csv. This can change from model to model.
- fn_col: *[str]* Column name of column containing full file paths to the images. Be aware that it must not start with "/", as it is automatically added. This can change from model to model.
- label_col: *[str]* Column name of column containing labels. In case *outcome* is unilabel, all unique occurrences are used as outcomes by converting the values to strings. If *outcome* is multilabel, then the column is converted to string and the cell values are split using *label_delim* and unique occurrences of these are created. If *outcome* is continuous, then the column is converted to float. This can change from model to model.
- valid_col: *[str]* Column name of column containing information whether the cases is used for tuning. Only accepted values in this column are 0 (training), 1 (tuning) of left empty (testing). The column is converted to a boolean before use. This can change from model to model.
- grad_cam_col: *[str]* Column name of column containing information whether a heatmap of activations, gradients of gradient-weighted class activation should be created for the given image. Only accepted values in this column are 0 (No), 1 (Yes) of left empty (No). This can change from model to model.

**IMAGE RESIZING**
- size: *[int, int]* Two integers defining the X and Y dimensions of the image to which they are rescaled. Minimum must be 224 for both dimensions.
- method: *[str]* The method used to achieve the required dimensions defined by size. Accepted values are: *squish* / *pad* / *crop*.
- pad_mode: *[str]* Method used to pad out missing pixels during resizing. Accepted values are: *zeros* / *border* / *reflection*.

**DATA AUGMENTATION**
- do_flip: *[1/0] * Should images be flipped horizontally for augmentation?
- flip_vert: *[1/0] * Should images be flipped vertically for augmentation?
- max_rotate: *[int]* Angle of maximum rotation for augmentation. Set to *0* for no rotation.
- min_zoom: *[float]* Minimum zoom ratio used for augmentation. Set to *1.0* for no zoom.
- max_zoom: *[float]* Maximum zoom ratio used for augmentation. Set to *1.0* for no zoom.
- max_warp: *[float]* Maximum warp applied to the images during augmentation. Set to *0* for no warp.
- p_affine: *[float]* The probability (number between 0.0 and 1.0) of affine (above) transformations.
- max_lighting: *[float]* Change brightness and contrast during augmentation.
- p_lighting: *[float]* The probability (number between 0.0 and 1.0) of lighting transformations.

**DL MODEL**
- arch: *[str]* Name of the deep learning architecture used. Can be one of the supported from: https://github.com/fastai/fastai/blob/master/fastai/vision/models/tvm.py.
- freeze_to: *[int]* Which parameter group to freeze model to. *-1* freezes body (conventional transfer learning using freeze()), *0* unfreezes all layers, other integers freeze up to that given parameter group or if the number is larger than the number of parameter groups then the whole model is frozen.
- cbs_n: *[function]* Callbacks defined as strings containing the given callback with all of its functions. Can add as many callbacks as the user wishes where n represents increasing numbers for each new callback. An example may be: CSVLogger(fname = att_dic['path'] + att_dic['model_dir'] + "/" + att_dic['model_f_name'] + "_log.csv", append = True). By default CSVLogger and SaveModelCallback are automatically added.

**DL TRAINING**
- loss_func: *[function]* The loss function supported by fastai (https://docs.fast.ai/losses.html) as a string. If *None* then the loss function is automatically selected based on the *label_col*.
- opt_func: *[function]* The optimizer function supported by fastai (https://docs.fast.ai/optimizer.html) as a string.
- moms: *[float, float, float]* Momentums used by the learning function.
- suggest_funcs *[str, str, str, str]* The name of the optimal learning rates found using *lr_find*. Supported values are: *minimum* / *steep* / *valley* / *slide*.
- lr_type: *[str]* Learning strategy. Currently *sequential* and *discriminative* are supported. In case of *sequential*, for all *suggest_funcs* the training does *n_epoch*, updating the learning rate using *lr_find* in each cycle. The best model resulting from any one of the cycles is saved at the end. For *discriminative*, a discriminative learning strategy is used by setting the learning rate to the minimum provided by *lr_find suggest_funcs* for the first layers and to the maximum provided by *lr_find suggest_funcs* for the last layers and training *n_epoch* times.
- bs: *[int]* The batch size used for training and validation.
- n_epoch: *[int]* The number of epochs for training.

**DL TEST**
- tta: *[1/0]* Should test-time augmentation be used for predictions?
- tta_n: *[int]* How many augmentations should be used for TTA.
- tta_beta: *[float]* With what weight should the augmented cases be considered in the final prediction using TTA.
- use_max: *[1/0]* Should the maximum be used instead of the average of the augmented and raw predictions in case of TTA?

**GradCAM**
- gradcam_dir: *[str]* Folder name within *path* where activation heatmaps will be saved. *_GradCAM_* and *model_f_name* are appended to the filename of the image.
- gradcam_outcome: *[int/str]* Which outcome to use to generate gradients. Either "truth", in which case the true label is used per image, or an integer specifying the ID of the outcome among possible IDs.
- model_layer: *[str]* Which layer of the model should be hooked if *model = learn.model.eval()*.
- sum_func: *[str]* Summary function to use on last layer of tensors, currently supported are *mean*, *median*, *max*, *mean_max*.
- sum_func: *[str]* Type of heatmap to use, currently supported are, *activation*, *gradient*, *activation_x_gradient*.
- sum_func: *[cmap]* Matplotlib color gradient to use.
- alpha: *[float]* Alpha level to use to superimpose color gradient.

**List of parameters needed for all tasks**
- *TASK SPECIFICATION*: load_existing, train, predict, grad_cam
- *GENERAL PARAMETERS*: outcome, label_delim, model_f_name, device, num_workers, model_dir
- *DATABASE PARAMETERS*: path, csv_fname, fn_col, label_col, valid_col
- *IMAGE RESIZING*: size, method, pad_mode
- *DATA AUGMENTATION*: do_flip, flip_vert, max_rotate, min_zoom, max_zoom, max_warp, p_affine, max_lighting, p_lighting
- *DL MODEL*: arch, cbs_n
- *DL TRAINING*: bs

**Additional parameters needed for training**
- *DL MODEL*: freeze_to
- *DL TRAINING*: loss_func, opt_func, moms, suggest_funcs, lr_type, n_epoch

**Additional parameters needed for predictions**
- *DL TEST*: tta, tta_n, tta_beta, use_max

**Additional parameters needed for activation heatmaps**
- *DATABASE PARAMETERS*: grad_cam_col
- *GradCAM*: gradcam_dir, gradcam_outcome, model_layer, sum_func, heatmap, cmap, alpha
