import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opendt", # Replace with your own username
    version="1.1",
    author="BAYANGMBE MOUNMO",
    author_email="bayangp0@gmail.com",
    description="Librairy to make a data augmmentation using tensorflow2.x and python3.x",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bm777/Data_augmentation",
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    classifiers=[
    'Development Status :: 4 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    "Operating System :: OS Independent",
  ],
)
