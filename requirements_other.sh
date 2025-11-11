pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118

wget https://download.pytorch.org/whl/cpu/torchdata-0.9.0%2Bcpu-cp39-cp39-linux_x86_64.whl#sha256=5656b119575c067fd48d4b52716666fb55f7bd643164c785124e67b23f376f94

wget https://data.pyg.org/whl/torch-2.3.0%2Bcu118/pyg_lib-0.4.0%2Bpt23cu118-cp39-cp39-linux_x86_64.whl

wget https://data.pyg.org/whl/torch-2.3.0%2Bcu118/torch_cluster-1.6.3%2Bpt23cu118-cp39-cp39-linux_x86_64.whl

wget https://data.pyg.org/whl/torch-2.3.0%2Bcu118/torch_scatter-2.1.2%2Bpt23cu118-cp39-cp39-linux_x86_64.whl

wget https://data.pyg.org/whl/torch-2.3.0%2Bcu118/torch_sparse-0.6.18%2Bpt23cu118-cp39-cp39-linux_x86_64.whl

wget https://data.pyg.org/whl/torch-2.3.0%2Bcu118/torch_spline_conv-1.2.2%2Bpt23cu118-cp39-cp39-linux_x86_64.whl

pip install *whl

rm *.whl

pip install -e .

cd ../ms-pred

git checkout iceberg_analychem_2024

pip install -e .