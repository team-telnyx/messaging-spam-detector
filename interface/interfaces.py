from abc import abstractmethod, ABCMeta
from typing import List

import attr
from attr.converters import optional
from attr.validators import instance_of


@attr.s
class SpamResponse:
    spam: List = attr.ib()
    error: List[str] = attr.ib()

class ISpamChecker(metaclass=ABCMeta):
    @abstractmethod
    async def filter_spam(self, *, user_id: str, msg_body: str) -> SpamResponse:
        """
        :param user_id:
        :param msg_body:
        :return:
        """
