from Chapter09.AlphaZero.mcts import MCTSPlayer
from Chapter09.AlphaZero.games.gomoku.game_ui import GameState as Gomoku
from Chapter09.AlphaZero.main import PipeLine

if __name__ == '__main__':

    board_size = 9
    n = 5

    pipeLine = PipeLine(board_size=board_size, n=n)
    pipeLine.load("models/best_model.pth")

    game_state = Gomoku(board_size, n)

    mcts_player = MCTSPlayer(policy=pipeLine.policy,
                             c_puct=5,
                             n_playout=1000,
                             is_self_play=False,
                             print_detail=True)
    while True:
        action = None
        if game_state.current_player == 1 and not game_state.pause:
            action = mcts_player.get_action(game_state.board)
        game_state.step(action)
