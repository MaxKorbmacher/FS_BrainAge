echo "We run all the FS7 model training"
echo "##############################"
echo "##############################"
echo "##############################"
echo "We start with SVM"
python3 FS7/brainage_training_SVM.py
echo "##############################"
echo "##############################"
echo "##############################"
echo "Now, we do XGB"
python3 FS7/brainage_training_XGB.py
echo "##############################"
echo "##############################"
echo "##############################"
echo "Now, we do Light GBM"
python3 FS7/brainage_training_LGBM.py
echo "##############################"
echo "##############################"
echo "##############################"
echo "Now, we do Lasso"
python3 FS7/brainage_training_Lasso.py
echo "##############################"
echo "##############################"
echo "##############################"
echo "All done"