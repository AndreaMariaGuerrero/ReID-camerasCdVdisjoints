How to use "ReID_CdVdisjoints_Superviseur"

PREREQUISITES :


- Matlab 2011b or later

- Gurobi Solver http://www.gurobi.com/LP

- SDALF code available on : https://github.com/lorisbaz/SDALF
**Person Re-identification by Symmetry-Driven Accumulation of Local Features**
M. Farenzena, L. Bazzani, A. Perina, V. Murino, and M. Cristani
*In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2010*

- ACF code available on : https://github.com/pdollar/toolbox
**Fast Feature Pyramids for Object Detection**
P. Dollar, R. Appel, S. Belongie, and P. Perona
*In IEEE Transactions on Pattern Analysis and Machine Intelligence, 2014*

- CamNet dataset available on : http://www.ee.ucr.edu/~amitrc/datasets.php
**A Camera Network Tracking (CamNeT) Dataset and Performance Baseline**
S. Zhang, E. Staudt, T. Faltemier, A. Roy-Chowdhury
*In IEEE Winter Conference on Applications of Computer Vision, 2015*



CODE :


"test_data_synchro.m" : apply ACF detector, build tracklet, apply k-means on each tracklet and calculate SDALF signature on an extract of CamNet dataset (3 cameras), and compute re-identification between any camera pair. We obtain a matrix for every camera pair corresponding to similarity score between individuals.

"test_data_synchro_4cam.m" : identical to the first program but add info of a 4th camera.

"superviseur.m" : apply superviseur of [1] on matrix of similarity score, with a constraint we added (minimum transition time between cameras). File in "Superviseur" directory.


other code '_'.m : Functions used in the two first script described here.
	- annotation.m : used to build ground truth of CamNet dataset
	- wrote_ids.m : Apply ground truth on our extract and record imagette for every detection (detection by ACF detector).
	- tracking.m : generate tracklets used to know time info.
	- apply_kmean.m : apply k-mean on tracklets for every cam
	- wHSVmatch_ajout3combi : distance computation between weighted HSV histogram. This distance is compute between every k detection of individuals of a camera pair.
	- crossvalidation_ajout3combi : adaptation of the distance matrix to the similarity matrix needed for our supervisor.
	- MSCRmatch_ajout3combi : distance computation between Maximum Stable Colour Region of the k detection of individuals of a camera pair. (It isn't used for now in our approach)
	- SetDataset_adapte.m : Adaptation of our dataset to apply SDALF code.
	
code '..._4cam'.m : Identical but used on the 4th camera.



[1] : **Consistent Re-identification In A Camera Network**. A. Das, A. Chakraborty, A. Roy-Chowdhury. *In European Conference on Computer Vision, pp. 330-345, vol.8690, Zurich, 2014*. 



USE : 


Create a directory "test_data_synchro" and "test_data_synchro_4cam" with subdirectories "imagette" and "mask". Masks are not provide in this version. You can create them with hand thanks to our code or use an other code to generate them.
Run "test_data_synchro.m" following instructions provided in the script and then run "test_data_synchro_4cam.m" if you want a 4th camera. Don't forget to record every time it is indicated if you want to use again the data.




