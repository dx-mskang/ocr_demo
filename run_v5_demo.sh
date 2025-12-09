DXRT_PATH=your/dx_rt/path

# Ensure virtual environment exists; if not, run startup.sh to create it
if [[ ! -d "venv" || ! -f "venv/bin/activate" || ! -d "engine/models/dxnn_optimized" || ! -d "engine/models/dxnn_mobile_optimized" ]]; then
    echo "Virtual environment or optimized model folders missing. Running ./startup.sh..."
    if [[ -x "./startup.sh" ]]; then
        ./startup.sh --dx_rt $DXRT_PATH
    else
        bash ./startup.sh --dx_rt $DXRT_PATH
    fi

    # verify creation
    if [[ ! -d "venv" || ! -f "venv/bin/activate" ]]; then
        echo "Failed to create virtual environment. Aborting." >&2
        exit 1
    fi
fi

source venv/bin/activate
source set_env.sh 1 2 1 3 2 4

# Prompt for using mobile model (default: no)
USE_MOBILE=""
read -t 10 -p "Use mobile model? [y/N] (auto-no in 10s): " USE_MOBILE || true

EXTRA_ARGS=""
if [[ "$USE_MOBILE" =~ ^[Yy]$ ]]; then
	EXTRA_ARGS="--use-mobile"
fi

python demo.py --version v5 $EXTRA_ARGS
