import cv2
import numpy

class Gomoku:

    def __init__(self, board_size = 9):
        self.board_size = board_size
        self.board = numpy.zeros((board_size, board_size))
        self.player = -1
        print(self.board)

    def playing_chess(self, x, y):
        if self.board[x, y] == 0:
            self.board[x, y] = self.player
            self.player = -self.player
            return True
        else:
            return False

    def check_win(self):
        is_win = False
        for i in range(self.board_size):
            for j in range(self.board_size):
                if is_win:
                    break
                '检查是否有连珠的情况'
                pieces_now = self.board[i, j]
                if pieces_now != 0 :
                    '检查横排的情况'
                    is_same = False
                    for count in range(1,5):
                        '''检查左边情况'''
                        if j <= 3:
                            break
                        else:
                            next_pieces = self.board[i,j-count]
                            if next_pieces == pieces_now:
                                is_same = True
                            else:
                                is_same = False
                                break
                    if is_same:
                        is_win = True
                        break

                    for count in range(1,5):
                        '''检查右边情况'''
                        if j >= self.board_size - 4:
                            break
                        else:
                            next_pieces = self.board[i,j+count]
                            if next_pieces == pieces_now:
                                is_same = True
                            else:
                                is_same = False
                                break
                    if is_same:
                        is_win = True
                        break
                    '''检查竖排情况'''
                    for count in range(1, 5):
                        '''检查上边情况'''
                        if i <= 3:
                            break
                        else:
                            next_pieces = self.board[i - count,j]
                            if next_pieces == pieces_now:
                                is_same = True
                            else:
                                is_same = False
                                break
                    if is_same:
                        is_win = True
                        break

                    for count in range(1, 5):
                        '''检查右边情况'''
                        if i >= self.board_size - 4:
                            break
                        else:
                            next_pieces = self.board[i + count,j]
                            if next_pieces == pieces_now:
                                is_same = True
                            else:
                                is_same = False
                                break
                    if is_same:
                        is_win = True
                        break

                    '''检查斜排情况'''
                    for count in range(1, 5):
                        '''检查左上情况'''
                        if j <= 3 and i <= 3:
                            break
                        else:
                            next_pieces = self.board[i - count, j - count]
                            if next_pieces == pieces_now:
                                is_same = True
                            else:
                                is_same = False
                                break
                    if is_same:
                        is_win = True
                        break

                    for count in range(1, 5):
                        '''检查右下边情况'''
                        if j >= self.board_size - 4 and i >= self.board_size - 4:
                            break
                        else:
                            next_pieces = self.board[i + count, j + count]
                            if next_pieces == pieces_now:
                                is_same = True
                            else:
                                is_same = False
                                break
                    if is_same:
                        is_win = True
                        break

                    for count in range(1, 5):
                        '''检查右上情况'''
                        if j >= self.board_size - 4 and i <= 3:
                            break
                        else:
                            next_pieces = self.board[i - count, j + count]
                            if next_pieces == pieces_now:
                                is_same = True
                            else:
                                is_same = False
                                break
                    if is_same:
                        is_win = True
                        break

                    for count in range(1, 5):
                        '''检查左下边情况'''
                        if j <= 3 and i >= self.board_size - 4:
                            break
                        else:
                            next_pieces = self.board[i + count, j - count]
                            if next_pieces == pieces_now:
                                is_same = True
                            else:
                                is_same = False
                                break
                    if is_same:
                        is_win = True
                        break
        return is_win


    def action(self,x,y):
        if self.playing_chess(x,y):
            return self.board, self.check_win()





if __name__ == '__main__':
    game = Gomoku()
    # game.playing_chess(4,4)
    # a = numpy.ones([2,2])
    # print(a[-2,-1])
    steps = [[0,1],[2,1],[0,2],[2,2],[0,3],[2,3],[0,4],[2,4],[0,5],[2,5]]
    for step in steps:
        a = input('input:')
        game.playing_chess(step[0],step[1])
        if game.check_win():
            print("win")
            break

