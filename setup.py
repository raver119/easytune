from distutils.core import setup
setup(
  name = 'easytune',
  packages = ['easytune'],
  version = '0.1',
  license='apache-2.0',
  description = 'Simple framework for hyperparameters optimization',
  author = 'Vyacheslav Kokorin',
  author_email = 'raver119@gmail.com',
  url = 'https://github.com/raver119/easytune',
  download_url = 'https://github.com/raver119/easytune/archive/v0.1.tar.gz',
  keywords = ['hyperparameters', 'hyperparams', 'optimization'],
  install_requires=[
          'numpy',
          'tensorflow>=2.0.0'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)