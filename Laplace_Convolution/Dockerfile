FROM continuumio/anaconda3:latest

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/workspace/src"

# Install necessary tools and libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libbz2-dev \
    libcfitsio-dev \
    zlib1g-dev \
    smbclient \
    libgl1-mesa-glx \
    cifs-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy the rest of the application code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    /opt/conda/bin/conda install jupyter -y --quiet

# Expose Jupyter Notebook port
EXPOSE 8888

# Create a non-root user and switch to it
RUN useradd -m vscode
# Change ownership of the workspace directory to the vscode user
RUN chown -R vscode:vscode /workspace
# Switch to the non-root user
USER vscode

# Default command to run when container starts
CMD ["/opt/conda/bin/jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
