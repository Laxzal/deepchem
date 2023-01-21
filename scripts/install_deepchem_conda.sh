# This script creates the new deepchem enviroment
# This script works on only Bash and Zsh

set -e # Exit if any command fails.

CMDNAME=`basename ${BASH_SOURCE:-$0}`
if [ $# -ne 2 ]; then
    echo "Please set two arguments."
    echo "Usage) source $CMDNAME python_version cpu_or_gpu" 1>&2
    echo "Example) source $CMDNAME 3.6 gpu" 1>&2
    return 1
fi

# This command is nearly equal to `conda init` command
# Need to use `conda activate` command
eval "$(conda shell.bash hook)"

# Create deepchem environment
mamba config --set always_yes yes
mamba create --name deepchem python=$1
mamba install -c conda-forge conda-merge

dir="$PWD/requirements"
if [ "$2" = "gpu" ];
then
    # We expect the CUDA vesion is 10.1.
    conda-merge $dir/env_common.yml $dir/torch/env_torch.gpu.yml $dir/env_test.yml $dir/jax/env_jax.gpu.yml > $PWD/env.yml
    echo "Installing DeepChem in the GPU environment"
else
    if [ "$(uname)" = 'Darwin' ]; then
        conda-merge $dir/env_common.yml $dir/env_mac.yml $dir/env_test.yml $dir/tensorflow/env_tensorflow.cpu.yml $dir/torch/env_torch.mac.cpu.yml $dir/jax/env_jax.cpu.yml > $PWD/env.yml
    elif [ "$(uname)" = 'Linux' ]; then
        conda-merge $dir/env_common.yml $dir/env_test.yml $dir/env_ubuntu.yml $dir/tensorflow/env_tensorflow.cpu.yml $dir/torch/env_torch.cpu.yml $dir/jax/env_jax.cpu.yml > $PWD/env.yml
    fi
    echo "Installing DeepChem in the CPU environment"
fi

# Install all dependencies
mamba env update --file $PWD/env.yml
