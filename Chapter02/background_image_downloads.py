import requests
import os
import re
import time


class Crawl_Pictures:

    def __init__(self, world, amount):
        self.world = world
        self.header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/83.0.4103.97 Safari/537.36 "
        }
        self.image_urls = None
        self.url = 'https://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result' \
                   '&fr=&sf=1&fmq' \
                   '=1591326388028_R&pv=&ic=0&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face' \
                   '=0&istype=2&ie' \
                   '=utf-8&sid=&word=' + self.world
        self.amount = amount
        self.counter = 1
        self.error_count = 0
        if not os.path.exists(self.world):
            os.mkdir(self.world)

    def __response(self):
        web_code = requests.get(self.url, headers=self.header)
        return web_code.text

    def __search_address(self, web_code):
        self.image_urls = re.findall('"thumbURL":"(.*?)","middleURL"', web_code)

    def next_page(self, web):
        next_url = re.findall('<a href="(.*?)" class="n">下一页</a>', web)
        next_url = "https://image.baidu.com" + next_url[0]
        self.url = next_url
        print("翻页成功")
        # print(self.url)

    def __download_img(self):
        for image_url in self.image_urls:
            try:
                image = requests.get(image_url, headers=self.header)
                file_name = str(self.counter) + image_url.split("/")[-1]
                print("[%d]正在下载图片：" % self.counter, file_name)
                with open(self.world + "/" + file_name, "wb") as f:
                    f.write(image.content)
                # time.sleep(0.5)
                self.counter += 1
            except Exception as e:
                print(e)
                self.error_count += 1
                pass
            if self.counter == self.amount:
                break

    def get_image(self):
        while True:
            web = self.__response()
            self.__search_address(web)
            self.__download_img()
            self.next_page(web)
            if self.counter == self.amount:
                print("下载完成")
                print("共下载图片%d张，出错%d次" % (self.counter, self.error_count))
                break


if __name__ == '__main__':
    key_word = input("请输入你要下载的图片：")
    amount = int(input("您一共需要多少张图片："))
    crawl = Crawl_Pictures(key_word,amount)
    crawl.get_image()
    print(crawl.url)
