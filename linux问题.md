## 查找文件
```bash
find / -name filename.txt
```
根据名称查找/目录下filename.txt文件

## 查看一个程序是否在运行

```bash
ps -ef | grep tomcat
``` 
查看所有有关tomcat的进程

## 终止线程

```bash
kill -9 19979
```
终止线程号位于19979的线程

## 查看文件，包含隐藏文件

```bash
ls -al
```

# 当前工作目录

```bash
pwd
```

## 复制文件包括其子文件到指定目录

```bash
cp -r sourceFolder targetFolder
```

## 创建目录

```bash
mkdir newFolder
```

## 删除目录（此目录是空目录）

```bash
rmdir deleteEmptyFolder
```

## 删除文件包括其子文件

```bash
rm -rf deleteFolder
```

## 移动文件

```bash
mv /temp/movefile /targetFolder
```

## 重命名

```bash
mv oldFileName newFileName
```

## 切换用户

```bash
su -username
```

## 修改文件权限

```bash
chmod 777 file.py # 修改后的权限为 rwxrwxrwx，rwx分别表示读、写、可执行
```

## 压缩文件

```bash
tar -czf test.tar.gz /test1 /test2
```

## 解压文件

```bash
tar -xvzf test.tar.gz
```

## 查看文件头10行

```bash
head -n 10 1.txt
```

## 查看文件最后10行

```bash
tail -n 10 1.txt
```

## 动态查看文件后10行

```bash
tail -f 1.txt
```

## 动态查看文件后100行

```bash
tail -100f 1.txt
```

## 查看系统当前时间

```bash
date
```
