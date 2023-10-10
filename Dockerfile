FROM waggle/plugin-base:1.1.1-ml

WORKDIR /app

RUN git clone https://github.com/grantwilkins/APPFL.git
RUN git checkout sz-compression
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install -e ".[dev,examples,analytics]"
RUN pip3 install zfpy scipy blosc zstd

CMD ["mpiexec -np 2", "python3", "./examples/cifar10.py"]
