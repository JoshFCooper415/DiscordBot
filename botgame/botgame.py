from flask import Flask, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import uuid
import os
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory game state
games = {}

class Player:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.prompts = []
        self.responses = []
        self.bot_responses = []
        self.score = 0
        self.votes_received = 0
        self.correctly_identified = 0

class Game:
    def __init__(self, id):
        self.id = id
        self.players = {}
        self.current_round = 0
        self.max_rounds = 3
        self.state = "waiting"
        self.current_voting_player = None
        self.votes = {}

    def add_player(self, player):
        self.players[player.id] = player

    def start_game(self):
        self.state = "playing"
        self.assign_prompts()

    def assign_prompts(self):
        prompts = [
            "Describe your perfect day",
            "What's your biggest fear?",
            "If you could have any superpower, what would it be?",
            "What's the most embarrassing thing that's ever happened to you?",
            "If you could travel anywhere in time, where and when would you go?",
        ]
        for player in self.players.values():
            player.prompts = random.sample(prompts, 3)

@app.route('/api/games', methods=['POST'])
def create_game():
    game_id = str(uuid.uuid4())
    games[game_id] = Game(game_id)
    return jsonify({"game_id": game_id}), 201

@app.route('/api/games/<game_id>/join', methods=['POST'])
def join_game(game_id):
    if game_id not in games:
        return jsonify({"error": "Game not found"}), 404
    
    data = request.get_json()
    player_name = data.get('name')
    player_id = str(uuid.uuid4())
    player = Player(player_id, player_name)
    games[game_id].add_player(player)
    
    return jsonify({"player_id": player_id, "game_id": game_id}), 200

@app.route('/api/games/<game_id>/start', methods=['POST'])
def start_game(game_id):
    if game_id not in games:
        return jsonify({"error": "Game not found"}), 404
    
    game = games[game_id]
    game.start_game()
    
    return jsonify({"message": "Game started"}), 200

@app.route('/api/games/<game_id>/players/<player_id>/prompts', methods=['GET'])
def get_prompts(game_id, player_id):
    if game_id not in games or player_id not in games[game_id].players:
        return jsonify({"error": "Game or player not found"}), 404
    
    player = games[game_id].players[player_id]
    return jsonify({"prompts": player.prompts}), 200

@app.route('/api/games/<game_id>/players/<player_id>/responses', methods=['POST'])
def submit_responses(game_id, player_id):
    if game_id not in games or player_id not in games[game_id].players:
        return jsonify({"error": "Game or player not found"}), 404
    
    game = games[game_id]
    player = game.players[player_id]
    data = request.get_json()
    player.responses = data.get('responses', [])
    
    # Here you would generate bot responses
    # For now, we'll use placeholder responses
    player.bot_responses = ["Bot response 1", "Bot response 2", "Bot response 3"]
    
    if all(len(p.responses) == 3 for p in game.players.values()):
        game.current_voting_player = next(iter(game.players))
        socketio.emit('start_voting', {'current_player': game.current_voting_player}, room=game_id)
    
    return jsonify({"message": "Responses submitted"}), 200

@app.route('/api/games/<game_id>/players/<player_id>/vote', methods=['POST'])
def submit_vote(game_id, player_id):
    if game_id not in games or player_id not in games[game_id].players:
        return jsonify({"error": "Game or player not found"}), 404
    
    game = games[game_id]
    data = request.get_json()
    game.votes[player_id] = data.get('vote', {})
    
    if len(game.votes) == len(game.players):
        # Calculate results
        # This is a simplified version, you'd need to implement the actual scoring logic
        for player in game.players.values():
            player.score = sum(1 for vote in game.votes.values() if vote.get('funniest') == player.id)
        socketio.emit('show_results', {'results': [{'name': p.name, 'score': p.score} for p in game.players.values()]}, room=game_id)
    else:
        game.current_voting_player = next(p for p in game.players if p != game.current_voting_player)
        socketio.emit('next_voting_player', {'current_player': game.current_voting_player}, room=game_id)
    
    return jsonify({"message": "Vote submitted"}), 200

@socketio.on('join')
def on_join(data):
    game_id = data['game_id']
    player_id = data['player_id']
    join_room(game_id)
    emit('player_joined', {'player_name': games[game_id].players[player_id].name}, room=game_id)

if __name__ == '__main__':
    socketio.run(app, debug=True)