# wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip -P domain_net_data
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip -P domain_net_data
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip -P domain_net_data
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip -P domain_net_data
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip -P domain_net_data
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip -P domain_net_data

wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt -P domain_net_data 
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt -P domain_net_data
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt -P domain_net_data
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt -P domain_net_data
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt -P domain_net_data
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt -P domain_net_data

# wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt -P domain_net_data
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt -P domain_net_data
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt -P domain_net_data
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt -P domain_net_data
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt -P domain_net_data
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt -P domain_net_data

unzip 'domain_net_data/*.zip'

# mv ../clipart domain_net_data
# mv infograph domain_net_data
# mv painting domain_net_data
# mv real domain_net_data

# mv quickdraw domain_net_data
# mv sketch domain_net_data
