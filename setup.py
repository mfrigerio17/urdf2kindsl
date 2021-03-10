import setuptools
import os, glob, shutil

with open("readme.md", "r") as fh:
    long_description = fh.read()


class RealClean(setuptools.Command):
    """Custom clean command to tidy up the project root."""
    CLEAN_FILES = './build ./dist ./*.egg-info */*.egg-info'.split(' ')

    user_options = []

    def initialize_options(self):
        pass
    def finalize_options(self):
        pass

    def run(self):
        for path_spec in RealClean.CLEAN_FILES:
            for path in glob.glob(path_spec):#[str(p) for p in abs_paths]:
                print('removing {0}'.format(path))
                shutil.rmtree(path)

setuptools.setup(
    name="urdf2kindsl",
    version="0.1.0",
    author="Marco Frigerio",
    author_email="marco.frigerio17@pm.me",
    description="URDF to KinDSL converter",
    long_description=long_description,
    long_description_content_type="text/markdown",

    packages = ['urdf2kindsl'],
    #package_dir = {'': 'urdf2kindsl'},

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.3',
    install_requires = [
        'numpy'
    ],

    entry_points = {
        "console_scripts": ["urdf2kindsl = urdf2kindsl.cmdline:main" ]
    },

    cmdclass={
        'realclean': RealClean,
    },
)



