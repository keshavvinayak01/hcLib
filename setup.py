from distutils.core import setup

setup(
    name='hcLib',  # Name of your library
    packages=['hcLib'],  # Automatically find and include all packages
    version='0.1.0',
    license='MIT',
    description='A comprehensive hybrid caching library for Python',  # Short description
    long_description='For more information on how to use, visit the repo at https://github.com/keshavvinayak01/hcLib',
    author='keshavvinayak01',
    author_email='keshavvinayakjha@gmail.com',
    url='https://github.com/keshavvinayak01/hcLib',
    keywords=['caching', 'data processing', 'augmentation', 'machine learning'],
    install_requires=[
        'cplex',
        'nvidia-dali-cuda120==1.31.0',
        'Pillow==10.0.1',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Information Technology',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
    ],
    setup_requires=['wheel'],
)
