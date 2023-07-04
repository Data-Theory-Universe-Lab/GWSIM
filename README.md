# GWSim

GWSim is a python code that allows to generate fake gravitational waves (GW) events.

# How-to install
* Clone the gwcosmo repository with
    ```
    git clone https://git.ligo.org/benoit.revenu/gwsim.git
    ```

* Set up a conda environment
  ```
  conda create -n gwsim python=3.9
  ```
  
* Activate your virtual environment
  ```
  conda activate gwsim
  ```
  
* Enter the cloned gwsim directory

* Install gwsim by running
    ```
    python -m pip install .
    ```

```
./bin/GW_create_universe --log_n -5
```