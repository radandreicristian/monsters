from src.vqa.base import BaseVqa
from src.vqa.blip import BlipVqa
from src.vqa.vilt import ViltVqa
from src.vqa.minicpm import MiniCpmVqa

class VqaFactory:

    mapping = {
        'blip': BlipVqa,
        'vilt': ViltVqa,
        'minicpm': MiniCpmVqa
    }
    
    def get_instance(self, name: str) -> BaseVqa:
        return self.mapping[name]()

if __name__ == '__main__':

    factory = VqaFactory()
    instance = factory.get_instance(name='blip')
    print(isinstance(instance, BlipVqa))