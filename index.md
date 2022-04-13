## What is HPC?
High performance computing (HPC) is often referred as **supercomputers** or **clusters**. A cluster is a pool of computing resources (e.g. CPUs, GPUs and Disk Drives) that can be allocated on demand for our computational needs. 

There are a number of clusters avaiable to UQ reserachers and studnets. Cluster users can submit jobs (code) to a cluster for execution on specified hardware. The computing resources on a cluster are shared among all cluster users. The resources and workload are managed by a central job management system (**slurm**). 

The typical workflow of running a computational task on a UQ hosted cluster involve:
1. Remotely access the login node of a UQ hosted cluster,
2. Copy the code over to the cluster,
3. Set up the necesary environements (e.g. pytorch) needed to execute the code, 
4. Send a request to the cluster workload managment system (**slurm**) to have your code executed,
5. Your code will be executed when the requested resources are available and it is your turn.

This guide will take you through the above steps on by setting up `pytorch` UQ's **Rangpur** cluster. Other clusters should work in a similar fashion.

*NOTE: First, make sure you are familiar with using [Unix Shell commands](https://swcarpentry.github.io/shell-novice/reference.html)* - most clusters don't have graphical interfaces.

## 1. Remotely Access the Login Node of a UQ's Rangpur Cluster
Use `ssh` to connect to and interact with the **login node** node of a Rangpur as follows 
```
ssh [user_name]@rangpur.compute.eait.uq.edu.au
```
Enter your password and this should put you in the `$HOME` directory. You can now interact with the cluster's login node by typing in Unix commands.

Enter `pwd` to see the absolute path of the current directory:
```
02:11:13 [user_name]@login1 ~ → pwd
/home/Staff/[user_name]
```

Enter `ls`, your current directory should be empty (for new users):
```
02:11:13 [user_name]@login1 ~ → ls
```

Leave the terminal window open.

*NOTE: If you are connecting to a UQ hosted cluster off compus, you will need to connect to [UQ's VPN](https://vpn.uq.edu.au) first.*


*NOTE: `ssh` is a native commnad on macOS and Linux. If you are on Windows, you can use [WSL](https://docs.microsoft.com/en-us/windows/wsl/install) or [Putty](https://www.putty.org) instead of `ssh`.*

## 2. Copy the Code Over to Rangpur
On your local computer, create a example script `main.py` using the editor of your choice. Pretend this is the training script of a neural network.
```
# main.py

import torch

# this should print "True" if the environment is configured correct with GPU access
print(torch.cuda.is_available()) 

```

Save the file and copy it over to the cluster using `scp [local file] [destination]`:
```
scp ./main.py [user_name]@rangpur.compute.eait.uq.edu.au:/home/Staff/[user_name]   
```
We have now copied the just created `main.py` to our remote `$HOME` directory (returned by the `pwd` command in step 1)

Enter `ls` again in the cluster terminal window, you should see the copy of `main.py` on Rangpur
```
02:14:04 [user_name]@login1 ~ → ls
main.py
```

Lets do a quick test by invoking the `python` command
```
02:28:04 [user_name]@login1 ~ → python main.py
-bash: python: command not found
```
**As you can see, the cluster does not have any enviroment set up to even run `python`, not to mention `pytorch`. We need to configure the enviroment oursevles for our script to run!**

## 3. Set Up The Necesary Environements on Rangpur
### 3.1 Installing Miniconda on Rangpur
To successfully execute `main.py`, we need to at least have `python` and `pytorch` configured on the cluster. Dependencies for deep learning are best managed using [conda](https://docs.conda.io/en/latest/miniconda.html), which can set up an entire GPU enabled `python` environment with a few lines.

However, `conda` is not to users of Rangpur by default as of early 2022.
```
02:38:05 [user_name]@login1 ~ → conda
-bash: conda: command not found
```
Luckily, `conda` can be easily installed. Open the [miniconda website](https://docs.conda.io/en/latest/miniconda.html), Copy the link of the package for `Miniconda3 Linux 64-bit`. Use `wget` on the cluster login node to download the installer to our `$HONE`.

```
02:41:16 [user_name]@login1 ~ → wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
--2022-04-13 14:41:21--  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
Resolving proxy.eait.uq.edu.au (proxy.eait.uq.edu.au)... 130.102.71.129
Connecting to proxy.eait.uq.edu.au (proxy.eait.uq.edu.au)|130.102.71.129|:8080... connected.
Proxy request sent, awaiting response... 200 OK
Length: 75660608 (72M) [application/x-sh]
Saving to: ‘Miniconda3-latest-Linux-x86_64.sh’

Miniconda3-latest-L 100%[===================>]  72.16M  41.9MB/s    in 1.7s

2022-04-13 14:41:24 (41.9 MB/s) - ‘Miniconda3-latest-Linux-x86_64.sh’ saved [75660608/75660608]

02:41:24 [user_name]@login1 ~ → ls
main.py  Miniconda3-latest-Linux-x86_64.sh
```

Run the installer, Press "Enter" to start installation:
```
02:42:13 [user_name]@login1 ~ → sh Miniconda3-latest-Linux-x86_64.sh
Welcome to Miniconda3 py39_4.11.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>
```
Agree to terms and conditions by entering `yes` when prompted:
```
Do you accept the license terms? [yes|no]
[no] >>> yes
```
Press "ENTER" to use `conda`'s preferred default location. All the `conda` binary will be stored here:
```
Miniconda3 will now be installed into this location:
/home/Staff/[user_name]/miniconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

```
`conda` will now fetch its essential dependencies. At the end, make sure you enter `yes` to initialise `conda`:

```
  ...
  zlib               pkgs/main/linux-64::zlib-1.2.11-h7f8727e_4


Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] >>> yes
```
*NOTE: this step is important! Your shell will not be able to locate the installed `conda` command if it is not initialised.*

You should see the following once the installation is complete:
```
==> For changes to take effect, close and re-open your current shell. <==

If you'd prefer that conda's base environment not be activated on startup,
   set the auto_activate_base parameter to false:

conda config --set auto_activate_base false

Thank you for installing Miniconda3!
``` 
As suggested, we should re-login to the cluster to see `conda` active. Enter `exit`, you will be disconnected from the cluster's login node. Use the same `ssh` command to reconnect:
```
02:52:34 [user_name]@login1 ~ → exit
logout
Connection to rangpur.compute.eait.uq.edu.au closed.

❯ ssh [user_name]@rangpur.compute.eait.uq.edu.au
[user_name]@rangpur.compute.eait.uq.edu.au's password:
```

Type `conda` again, you should see the following (no longer `command not found`):
```
02:53:52 [user_name]@login1 ~ → conda
usage: conda [-h] [-V] command ...

conda is a tool for managing and deploying applications, environments and packages.

Options:

positional arguments:
  command
    clean        Remove unused packages and caches.
    compare      Compare packages between conda environments.
    config       Modify configuration values in .condarc. This is modeled after the git config command. Writes to the user .condarc file (/home/Staff/[user_name]/.condarc) by
                 default.
    create       Create a new conda environment from a list of specified packages.
    help         Displays a list of available conda commands and their help strings.
    info         Display information about current conda install.
    init         Initialize conda for shell interaction. [Experimental]
    install      Installs a list of packages into a specified conda environment.
    list         List linked packages in a conda environment.
    package      Low-level conda package utility. (EXPERIMENTAL)
    remove       Remove a list of packages from a specified conda environment.
    uninstall    Alias for conda remove.
    run          Run an executable in a conda environment. [Experimental]
    search       Search for packages and display associated information. The input is a MatchSpec, a query language for conda packages. See examples below.
    update       Updates conda packages to the latest compatible version.
    upgrade      Alias for conda update.

optional arguments:
  -h, --help     Show this help message and exit.
  -V, --version  Show the conda version number and exit.

conda commands available from other packages:
  content-trust
  env
```
[Optinal] We may now remove the installer:
```
02:57:04 [user_name]@login1 ~ → rm Miniconda3-latest-Linux-x86_64.sh
```

### 3.2 Installing a `pytorch` Envornment Using Miniconda
We will now use `conda` to install a **self-contained** python virtual enviroment containing `pytorch`. In other words, we will ask `conda` to fetch everything we need to run `pytorch` code.

Lets create a virtual enviroment to store the dependecies we need:
```
conda create --prefix ./my-env pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
Breakdown: 
- `conda create` creates a new virtual enviroment.
- `--prefix` allows us to specific a specific directory to story the downloaded dependencies. Refer to `conda`'s documentation for other options.
- `./my-env` is the directory I specified to store the packages used by this enviroment, it can be any director you have access to.
- `pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch` are the packages we need to download to get `pytorch` running on GPU. Others have already done the hard work building these packages so installing everything is a one-liner for us!

`y` to confirm when prompted:
```
  ...
  xz                 pkgs/main/linux-64::xz-5.2.5-h7b6447c_0
  zlib               pkgs/main/linux-64::zlib-1.2.11-h7f8727e_4
  zstd               pkgs/main/linux-64::zstd-1.4.9-haebb681_0


Proceed ([y]/n)? y
```

The installation of this virual envorment should take around 10 mins and you will see the following once done:
```
done
#
# To activate this environment, use
#
#     $ conda activate /home/Staff/[user_name]/my-env
#
# To deactivate an active environment, use
#
#     $ conda deactivate

```

Enter `ls`, you should see the directory (`my-env`) we instructed `conda` to store enviroment dependencies in:
```
03:12:14 [user_name]@login1 ~ → ls
main.py  miniconda3  Miniconda3-latest-Linux-x86_64.sh  my-env
```
Up until this point, the enviroment is still not active so the packages in it are still not accessible to us. We will need to activate the environment next.

### 3.3 Activating the `pytorch` Envornment
`conda` created enviroments will persist on the cluster, we just need to activate them before we use them, no need to reinstall every time we log in. Any `conda` environment install using the `--prefix` flag can be activated using `conda activate [env-dir]`. In this case, type: 
```
conda activate /home/Staff/[user_name]/my-env
```
Finally, we do a quick test using our `main.py` to see if it works
```
03:16:11 [user_name]@login1 ~ → python main.py
True
```
The script prints `True` indicating pytorch is indeed installed and it is running on GPU. We can theoretically run nerual networks on GPU now!

However, we are not done yet! 

<!-- 
You can use the [editor on GitHub](https://github.com/SiyuLiu0329/uq-hpc-guide2022/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/SiyuLiu0329/uq-hpc-guide2022/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out. -->
