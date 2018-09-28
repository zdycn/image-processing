import os
import random
import shutil
from PIL import Image
from PIL import ImageFile
# import data_utils1
import xlrd
ImageFile.LOAD_TRUNCATED_IMAGES = True


def append_class_label_on_filename(source_path, save_path) :
    """
    在每个图像文件名前面加入类标签
    比如：source_path="g:/kaggle"，kaggle文件夹下有5类（0,1,2,3,4）
    每个子文件夹下的文件名根据类别加入前缀（例如：10_left.jpeg变为0_10_left.jpeg)

    For example：
        append_class_label_on_filename("g:/kaggle", "g:/kaggle-append-class-label")

    :param source_path:
    :param save_path:
    """
    def change_name_by_dirName(file_path,save_path,class_name):
        newDir = os.path.join(save_path, class_name)
        if(not os.path.exists(save_path)):
            os.mkdir(save_path)
        if( not os.path.exists(newDir) ):
            os.mkdir(newDir)
        filelist = os.listdir(file_path)  # 该文件夹下所有的文件（包括文件夹）
        for files in filelist:
            oldName = os.path.join(file_path, files)
            newName = os.path.join(newDir, class_name+'_'+files)
            shutil.copy(oldName, newName)
    filelist=os.listdir(source_path)#该文件夹下所有的文件（包括文件夹）
    for files in filelist:#遍历所有文件
        print( files)
        Olddir=os.path.join(source_path,files)#原来的文件路径
        if os.path.isdir(Olddir) and( Olddir != save_path ) :#如果是文件夹则跳过
                change_name_by_dirName( Olddir,save_path,files )

# TODO 该函数已经过时
def get_misclassified_images(experiment_results, images_path, misclassified_images_path) :
    """
    根据实验结果文件（experiment_results）把所有误分类的图像文件保存到指定的文件夹（misclassified_images_path）
    文件夹中包含两个子文件夹：0-1（实际类别0误分为1）和1-0（实际类别1误分为0）

    For example：
        get_misclassified_images("experiment.xls", "g:/kaggle", "g:/kaggle-misclassified-images")

    :param experiment_results: excel文件，文件中列出了所有图像的分类结果
    :param images_path: 所有图像（测试样本）的路径
    :param misclassified_images_path: 所有误分类图像的保存路径
    """
    xlsfile = experiment_results  # 打开指定路径中的xls文件
    readbook = xlrd.open_workbook(xlsfile)  # 得到Excel文件的book对象，实例化对象
    sheet0 = readbook.sheet_by_index(0)  # 通过sheet索引获得sheet对象
    nrows = sheet0.nrows  # 获取行总数
    # #循环打印每一行的内容
    if (not os.path.exists(misclassified_images_path)):
        os.mkdir(misclassified_images_path)
    for i in range(nrows):
        img = sheet0.row_values(i)[0]
        if (sheet0.row_values(i)[1] != sheet0.row_values(i)[2] and '.' in img):
            oldClass = str(int(sheet0.row_values(i)[1]))
            newClass = str(int(sheet0.row_values(i)[2]))
            print(oldClass, newClass)
            newPath = misclassified_images_path + '\\' + oldClass + "-" +newClass
            if (not os.path.exists(newPath)):
                os.mkdir(newPath)
            if (os.path.exists(images_path + '\\' + oldClass + '\\' + img)):
                shutil.copy(images_path + '\\' + oldClass + '\\' + img, newPath + '\\'+img )
            else:
                print('指定文件夹没有' + img)

# TODO 代码修改： data_utils是tflearn的函数，需要修改为Keras，数据集保存存hdf567格式
def make_dataset(indexfiles, class_num, image_paths, dataset_paths,trainset_nums, validationset_nums, testset_nums) :
    """
    在清理后的图像中，构造训练集、验证集、测试集，生成三种数据集的pkl文件

    通过随机采样构造样本集：按照每一类占总的图像集的比例，进行采样；如果某一类太少，就提取全部样本
    （比如：严重疾病类只有5个样本，就取所有样本）！！！

    For example:
        make_dataset("g:/kaggle_datacleaning.txt",10, "g:/kaggle", "g:/kaggle-data",1000, 200, 200)

    :param indexfiles: 文本文件，包含所有已完成数据筛选的图像文件名
    :param class_num:样本包含类数
    :param image_paths: 所有已完成数据筛选的图像所在路径
    :param dataset_paths: 数据集的图像及pkl存放路径
    :param trainset_nums: 训练集样本数
    :param validationset_nums: 验证集样本数
    :param testset_nums: 测试集样本数
    :return:
    """

    def make_dataset( dataset,dataset_num,class_rate,image_paths,dataset_paths ):
        index = 0
        if (not os.path.exists(dataset_paths)):
            os.mkdir(dataset_paths)
        for rate in class_rate:
            if( int(rate*dataset_num)<1 ):
                img_num = 1
            else:
                img_num = int(rate*dataset_num)
            image_path = image_paths+'/'+str(index)
            dataset_path = dataset_paths+'/'+str(index)
            if dataset[index].__len__() < img_num:
                img_num = dataset[index].__len__()
            newDataset = random.sample(dataset[index], img_num)
            if (not os.path.exists(dataset_path)):
                os.mkdir(dataset_path)
            for img in newDataset:
                dataset[index].remove( img )
                shutil.copy(image_path+'/'+img, dataset_path+'/'+img)
            index += 1
        return dataset
    img_number = [0]*class_num #存放每类图像的数目
    class_rate = [] #存放每类图像所占比例
    imgs = [[]for i in range(class_num)] #分别存放各类图像
    fd = open(indexfiles)
    if (not os.path.exists(dataset_paths)):
        os.mkdir(dataset_paths)
    for line in fd.readlines():
        img_class = int(line[0])
        img_number[img_class] +=1
        if( '\n' in line ):
            imgs[ img_class ].append( line[:-1] )

        else:
            imgs[img_class].append(line)
    fd.close()
    imgs_sum = 0
    for num in img_number:
        imgs_sum +=num
    for num in img_number:
        class_rate.append( num/imgs_sum )
    imgs = make_dataset( dataset=imgs, dataset_num=testset_nums, class_rate=class_rate, image_paths=image_paths, dataset_paths=dataset_paths+'/test')
    imgs = make_dataset(dataset=imgs, dataset_num=validationset_nums, class_rate=class_rate, image_paths=image_paths,
                        dataset_paths=dataset_paths + '/validation')
    make_dataset(dataset=imgs, dataset_num=trainset_nums, class_rate=class_rate, image_paths=image_paths,
                        dataset_paths=dataset_paths + '/train')
    data_utils.build_image_dataset_from_dir_no(dataset_paths+'/train',
                                                dataset_file=dataset_paths+"/train.pkl",
                                                filetypes=['.jpg', '.jpeg'],
                                                convert_gray=False,
                                                shuffle_data=True,
                                                categorical_Y=True)
    data_utils.build_image_dataset_from_dir_no(dataset_paths+'/test',
                                                dataset_file=dataset_paths+"/test.pkl",
                                                filetypes=['.jpg', '.jpeg'],
                                                convert_gray=False,
                                                shuffle_data=True,
                                                categorical_Y=True)
    data_utils.build_image_dataset_from_dir_no(dataset_paths+'/validation',
                                                dataset_file=dataset_paths+"/validation.pkl",
                                                filetypes=['.jpg', '.jpeg'],
                                                convert_gray=False,
                                                shuffle_data=True,
                                                categorical_Y=True)


# TODO 修改为用scikit-learn来随机采样
def sample(images_file, samples_file, sample_nums):
    """
    从图像集中随机抽取指定数量的样本

    For example：
        sample("g:/kaggle-0.txt", "g:/kaggle-0-train.txt", 50)
        从kaggle-0.txt中随机抽取50个图像，图像名保存到 kaggle-0-train.txt
    :param images_file: txt文件，文件中包含所有图像的文件名
    :param samples_file: txt文件，文件中包含采样图像的文件名
    :param sample_nums: 采样数
    :return: 所有采样的图像文件名
    """
    results = []
    fd = open(images_file)
    for line in fd.readlines():
        results.append(line)

    slice = random.sample(results, sample_nums)
    fr = open(samples_file, 'w')
    for i in slice:
        fr.write(i)
    fd.close()
    fr.close()

# TODO 考虑修改文件名
def get_indexfile_from_images(indexfile, images_path):
    """
    根据图像所在路径创建索引文件（包含图像路径中所有的文件名）

    For example：
        get_indexfile_from_images("g:/kaggle-datacleaning.txt", "g:/kaggle-datacleaning") :

    :param indexfile: txt文件，包含图像文件名
    :param images_path: 所有图像的存放路径
    """
    dirs = os.listdir(images_path)
    f = open(indexfile, 'w')
    for i in dirs:
        if os.path.splitext(i)[1] == ".jpg" \
                or os.path.splitext(i)[1] == ".jpeg":  # 筛选出图片文件
            result = i + '\n'
            f.write(result)
    f.close()

# TODO 考虑修改文件名
def get_images_from_indexfile(indexfile, images_path, sample_path):
    """
    根据索引文件（包含图像文件名）提取对应的图像
    For example：
        get_images_from_indexfile("g:/kaggle-datacleaning.txt", "g:/kaggle-datacleaning", "g:/kaggle-datacleaning-train") :

    :param indexfile: txt文件，包含图像文件名
    :param images_path: 所有图像的存放路径
    :param sample_path: 提取图像的存放路径
    """
    if os.path.exists(sample_path):
        # os.remove(path_des) # 系统权限不够，需要手动删除
        pass
    else:
        os.makedirs(sample_path)

    results = []
    dirs = os.listdir(images_path)
    fd = open(indexfile)
    for line in fd.readlines():
        results.append(line)
    for i in dirs:
        if (i + '\n') in results:
            shutil.move(images_path + '/' + i, sample_path + '/' + i)

def crop(src_path,dst_path,revised_size):
    """
    从源图像集中读取所有图片文件，将其复制到目标路径并重新设定图片的分辨率（裁剪图片）

    For example：
        crop("g:/kaggle", "g:/kaggle-new", 256)
        从kaggle中读取所有图片文件，在kaggle——new中存储其分辨率为256*256版
    :param src_path: 需要调整分辨率的图片路径
    :param dst_path: 调整后图片的存储路径
    :param revised_size: 要调整的分辨率大小
    """

    def getAllImages(path):
        """
        获得图片的绝对路径
        :param path: 目标图片的路径
        :return: 图片的绝对路径
        """
        return [os.path.join(path, f) for f in os.listdir(path) if
                f.endswith('.jpg') | f.endswith('jpeg') | f.endswith('tif')]

    def copyFiles(src_Path, dst_Path):
        """
        将源路径文件复制到目标路径
        :param src_Path: 源文件路径
        :param dst_Path: 目标路径
        :return: None
        """
        if not os.path.exists(src_Path):
            print("src path not exist!")
            exit()
        if not os.path.exists(dst_Path):
            os.makedirs(dst_Path)
        for root, dirs, files in os.walk(src_Path):
            for eachfile in files:
                shutil.copy(os.path.join(root, eachfile), dst_Path)

    copyFiles(src_path, dst_path)

    for path in getAllImages(dst_path):
        # 图片复制后所在路径
        im = Image.open(path)
        if len(im.split()) == 4:
            r, g, b, a = im.split()
            im = Image.merge("RGB", (r, g, b))
        x, y = im.size
        if x > y:
            z = (x - y) / 2
        else:
            z = (y - x) / 2
        if x > y:
            box = (z, 0, y + z, y)
        else:
            box = (0, z, x, z + x)
        im1 = im.crop(box)
        # 裁剪
        im2 = im1.resize((revised_size, revised_size))
        # 缩小
        im2.save(path)
    print("Finished")


# sample("G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-test\\已选\\kaggle-test-已选4.txt","G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-test\\已选\\test4(20180718).txt",40)

# get_images_from_indexfile("G:\\医院训练集图片医院图\\kaggle-全\\增量-0815\\4.txt","G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-train\\已选\\4","G:\\医院训练集图片医院图\\kaggle-全\\增量-0815\\4")

# crop("F:\\测试暂用空间\\问题图片","F:\\测试暂用空间\\问题图片new",512)

# get_misclassified_images("F:\\测试暂用空间\\wrong.xlsx","F:\\测试暂用空间\\yiyuan--test","F:\\测试暂用空间\\wrong_img")

# get_indexfile_from_images("G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-train\\已选\\0.txt","G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-train\\已选\\0")
# get_indexfile_from_images("G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-train\\已选\\1.txt","G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-train\\已选\\1")
# get_indexfile_from_images("G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-train\\已选\\2.txt","G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-train\\已选\\2")
# get_indexfile_from_images("G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-train\\已选\\3.txt","G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-train\\已选\\3")
# get_indexfile_from_images("G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-train\\已选\\4.txt","G:\\医院训练集图片医院图\\kaggle-全\\已清洗\\kaggle-train\\已选\\4")
# append_class_label_on_filename("F:\\测试暂用空间\\分类测试数据","F:\\测试暂用空间\\医院分类测试数据（已命名）")
crop("f:\\1","F:\\0", 512)
