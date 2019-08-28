import os
import tarfile


def make_tarball(base_name, root_dir):
    """
    将``root_dir``下的文件打包为一个tar文件

    base_name: 打包后的文件名为``base_name``.tar
    root_dir: 需要打包的文件所在的目录
    """
    with tarfile.open('.'.join([base_name, 'tar']), 'w') as f:
        for file in os.listdir(root_dir):
            f.add(os.path.join(root_dir, file), arcname=file)
