/**
 * High-performance Connect-4 Bitboard Alpha-Beta Solver with Transposition Table
 * Solves positions and outputs scores for all legal columns.
 * Column indices are 0-6.
 * Output format per line:
 * <col0_score> <col1_score> ... <col6_score>
 * If column is illegal or instant loss/win, score reflects distance to win/loss or DRAW (0).
 */

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <unordered_map>

using namespace std;

static const int WIDTH = 7;
static const int HEIGHT = 6;
static const int MIN_SCORE = -(WIDTH * HEIGHT) / 2 + 3;
static const int MAX_SCORE = (WIDTH * HEIGHT + 1) / 2 - 3;

class Position {
public:
    uint64_t current_position;
    uint64_t mask;
    int moves;

    Position() : current_position(0), mask(0), moves(0) {}

    static uint64_t bottom_mask(int col) {
        return UINT64_C(1) << (col * (HEIGHT + 1));
    }

    static uint64_t column_mask(int col) {
        return ((UINT64_C(1) << HEIGHT) - 1) << (col * (HEIGHT + 1));
    }

    bool can_play(int col) const {
        return (mask & (UINT64_C(1) << (col * (HEIGHT + 1) + HEIGHT - 1))) == 0;
    }

    void play(int col) {
        current_position ^= mask;
        mask |= mask + bottom_mask(col);
        moves++;
    }

    bool is_winning_move(int col) const {
        uint64_t pos = current_position;
        pos |= (mask + bottom_mask(col)) & column_mask(col);
        return alignment(pos);
    }

    static bool alignment(uint64_t pos) {
        // Horizontal
        uint64_t m = pos & (pos >> 7);
        if (m & (m >> 14)) return true;
        // Diagonal \
        m = pos & (pos >> 6);
        if (m & (m >> 12)) return true;
        // Diagonal /
        m = pos & (pos >> 8);
        if (m & (m >> 16)) return true;
        // Vertical
        m = pos & (pos >> 1);
        if (m & (m >> 2)) return true;
        return false;
    }

    uint64_t key() const {
        return current_position + mask;
    }
};

struct TTEntry {
    uint64_t key;
    int8_t val;
};

static const int TT_SIZE = 8388608; // 8M entries
static vector<TTEntry> TT(TT_SIZE, {0, 0});

static const int column_order[7] = {3, 2, 4, 1, 5, 0, 6};

int negamax(const Position &pos, int alpha, int beta, int depth) {
    if (pos.moves >= WIDTH * HEIGHT) return 0; // Draw

    for (int c = 0; c < WIDTH; ++c) {
        if (pos.can_play(c) && pos.is_winning_move(c)) {
            return (WIDTH * HEIGHT + 1 - pos.moves) / 2;
        }
    }

    int max_possible = (WIDTH * HEIGHT - 1 - pos.moves) / 2;
    if (beta > max_possible) {
        beta = max_possible;
        if (alpha >= beta) return beta;
    }

    uint64_t k = pos.key();
    int tt_idx = k % TT_SIZE;
    if (TT[tt_idx].key == k) {
        return TT[tt_idx].val;
    }

    for (int i = 0; i < WIDTH; ++i) {
        int col = column_order[i];
        if (pos.can_play(col)) {
            Position next_pos = pos;
            next_pos.play(col);
            int score = -negamax(next_pos, -beta, -alpha, depth + 1);
            if (score >= beta) {
                TT[tt_idx] = {k, static_cast<int8_t>(score)};
                return score;
            }
            if (score > alpha) alpha = score;
        }
    }

    TT[tt_idx] = {k, static_cast<int8_t>(alpha)};
    return alpha;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    string line;
    while (getline(cin, line)) {
        if (line.empty()) continue;
        // Input: string of moves played, e.g. "3324..." or sequence of col numbers 0-6 or 1-7
        Position pos;
        bool valid = true;
        for (char ch : line) {
            int col = -1;
            if (ch >= '0' && ch <= '6') col = ch - '0';
            else if (ch >= '1' && ch <= '7') col = ch - '1';
            if (col >= 0 && col < WIDTH) {
                if (pos.can_play(col)) {
                    pos.play(col);
                } else {
                    valid = false;
                    break;
                }
            }
        }

        if (!valid) {
            cout << "INVALID" << "\n";
            continue;
        }

        // Evaluate all legal columns
        vector<int> scores(WIDTH, -999);
        for (int c = 0; c < WIDTH; ++c) {
            if (!pos.can_play(c)) {
                scores[c] = -999;
            } else if (pos.is_winning_move(c)) {
                scores[c] = (WIDTH * HEIGHT + 1 - pos.moves) / 2;
            } else {
                Position next_pos = pos;
                next_pos.play(c);
                scores[c] = -negamax(next_pos, -25, 25, 0);
            }
        }

        for (int c = 0; c < WIDTH; ++c) {
            cout << scores[c] << (c == WIDTH - 1 ? "" : " ");
        }
        cout << "\n" << flush;
    }
    return 0;
}
