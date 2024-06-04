INSTALLATION 

1 - Create a dedicated conda environment to avoid package conflicts:

conda create --name wlcl_env

2 - Go in the environment, install newest version of pymc:

conda install -c conda-forge pymc

4 - Install wlclusters:

git clone https://github.com/LorisChappuis/wlclusters.git
cd wlclusters
pip install .

3 - Adding the environment as a jupyter notebook kernel :

conda install -c anaconda ipykernel
python -m ipykernel install --user --name=wlcl_env


4 - install ipywidgets for better pymc visualisation in jupyter-notebook (optional)

conda install ipywidgets
