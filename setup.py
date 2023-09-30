from setuptools import setup, find_packages

setup(
    name="Whisper Streaming",
    version="1.0.0",
    packages=find_packages(),  # Replace with your actual package name
    install_requires=[
        "faster-whisper",  # Replace with the name of your dependency
    ],
    author="Vikentii Pankov",
    author_email="vkpankov@email.com",
    description="Whisper for processing large files and audio streams",
    url="https://github.com/yourusername/your_project",
    license="MIT",  # You can specify your project's license
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
