#ifndef RELEARN_WINDOW_CHIPTUNE_HPP
#define RELEARN_WINDOW_CHIPTUNE_HPP
/**
 * The lessons' music and sound effects, written as code in the manner of an old games console: two pulse waves, a
 * triangle wave and noise. There are no sound files, and no raylib here: everything is an array of samples, which the
 * tests can measure. The tune is original.
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace games
{
namespace lessons
{
/// samples per second of everything in this file
constexpr int sample_rate = 44100;

/**
 * @class waveform
 * @brief the shape of a voice
 */
class waveform
{
public:
    virtual ~waveform() = default;
    /// @return the voice at @param phase, from 0 up to 1 along one cycle
    virtual double at(double phase) = 0;
    /// @brief forget the past: every note starts the same way
    virtual void restart() {}
};

/**
 * @class pulse_wave
 * @brief the buzzing square voice; its average is zero, so that it adds no thump
 */
class pulse_wave : public waveform
{
public:
    /// @param duty is the part of a cycle spent high
    explicit pulse_wave(double duty)
    : duty_(duty)
    {}

    double at(double phase) override
    {
        return phase < duty_ ? 1 - duty_ : -duty_;
    }

private:
    double duty_;
};

/**
 * @class triangle_wave
 * @brief the soft voice of the bass; it starts from zero
 */
class triangle_wave : public waveform
{
public:
    double at(double phase) override
    {
        const double shifted = std::fmod(phase + 0.25, 1.0);
        return shifted < 0.5 ? 4 * shifted - 1 : 3 - 4 * shifted;
    }
};

/**
 * @class noise_wave
 * @brief the voice of the drums: a 15-bit shift register, changing once a cycle
 */
class noise_wave : public waveform
{
public:
    double at(double phase) override
    {
        if (phase < last_) {
            const unsigned int feedback = (bits_ ^ (bits_ >> 1)) & 1u;
            bits_ = (bits_ >> 1) | (feedback << 14);
        }
        last_ = phase;
        return (bits_ & 1u) ? 0.5 : -0.5;
    }

    void restart() override
    {
        bits_ = 1;
        last_ = 0;
    }

private:
    unsigned int bits_ = 1;
    double last_ = 0;
};

/**
 * @struct envelope
 * @brief how the loudness of a note moves: up at once, down to the sustain, and to silence at its end
 */
struct envelope
{
    /// seconds to full loudness
    double attack = 0.003;
    /// time constant of the fall to the sustain, in seconds
    double decay = 0.12;
    /// the loudness held after the fall, from 0 to 1
    double sustain = 0.55;
    /// seconds to silence, at the very end of the note
    double release = 0.012;

    /// @return the loudness at @param t seconds into a note of @param length seconds
    double level(double t,
                 double length) const
    {
        if (t < 0 || t >= length) {
            return 0;
        }
        const double rise = std::min(1.0, t / attack);
        const double body = t < attack ? 1.0 : sustain + (1 - sustain) * std::exp(-(t - attack) / decay);
        const double fall = std::min(1.0, (length - t) / release);
        return rise * body * fall;
    }
};

/// @return the frequency in Hz of a note such as "A4", "C#5" or "Bb3"
inline double frequency_of(const std::string & name)
{
    static const std::string letters = "CDEFGAB";
    static const int semitones[7] = {0, 2, 4, 5, 7, 9, 11};
    const std::size_t letter = name.empty() ? std::string::npos : letters.find(name[0]);
    const bool sharp = name.size() == 3 && name[1] == '#';
    const bool flat  = name.size() == 3 && name[1] == 'b';
    const bool plain = name.size() == 2;
    if (letter == std::string::npos || !(sharp || flat || plain) || !std::isdigit(static_cast<unsigned char>(name.back()))) {
        throw std::invalid_argument("not a note: " + name);
    }
    const int octave = name.back() - '0';
    const int from_a4 = semitones[letter] + (sharp ? 1 : 0) - (flat ? 1 : 0) + 12 * (octave - 4) - 9;
    return 440.0 * std::pow(2.0, from_a4 / 12.0);
}

/**
 * @struct note
 * @brief a note of a score, placed in steps
 */
struct note
{
    double frequency;
    std::size_t start;
    std::size_t length;
};

/**
 * @class score
 * @brief one voice, written one token per step: a note ("E5"), "-" to hold the last one, "." for silence
 */
class score
{
public:
    /// @param text is the voice's tokens, separated by spaces
    explicit score(const std::string & text)
    {
        std::istringstream tokens(text);
        std::string token;
        bool holding = false;
        while (tokens >> token) {
            if (token == "-") {
                if (holding) {
                    notes_.back().length++;
                }
            }
            else if (token == ".") {
                holding = false;
            }
            else {
                notes_.push_back({frequency_of(token), steps_, 1});
                holding = true;
            }
            steps_++;
        }
    }

    /// @return the notes, in order
    const std::vector<note> & notes() const
    {
        return notes_;
    }

    /// @return how many steps it lasts
    std::size_t steps() const
    {
        return steps_;
    }

private:
    std::vector<note> notes_;
    std::size_t steps_ = 0;
};

/**
 * @class mixer
 * @brief adds voices up into one track of a fixed length
 */
class mixer
{
public:
    /// @param samples is the length of the track
    explicit mixer(std::size_t samples)
    : track_(samples, 0.0)
    {}

    /**
     * @brief add a note whose pitch slides, in equal ratios, from @param from to @param to
     * @param voice is its shape
     * @param from and @param to are in Hz (equal for a plain note)
     * @param start and @param seconds place it in the track
     * @param shape is its loudness over time
     * @param level multiplies it
     */
    void slide(waveform & voice,
               double from,
               double to,
               double start,
               double seconds,
               const envelope & shape,
               double level)
    {
        voice.restart();
        const auto first = static_cast<std::size_t>(start * sample_rate);
        const auto count = static_cast<std::size_t>(seconds * sample_rate);
        double phase = 0;
        for (std::size_t i = 0; i < count && first + i < track_.size(); i++) {
            const double t = static_cast<double>(i) / sample_rate;
            track_[first + i] += level * shape.level(t, seconds) * voice.at(phase);
            phase = std::fmod(phase + from * std::pow(to / from, t / seconds) / sample_rate, 1.0);
        }
    }

    /// @brief add every note of @param part, each step lasting @param step_seconds
    void play(waveform & voice,
              const score & part,
              double step_seconds,
              const envelope & shape,
              double level)
    {
        for (const auto & n : part.notes()) {
            slide(voice, n.frequency, n.frequency, n.start * step_seconds, n.length * step_seconds, shape, level);
        }
    }

    /// @return the track as 16-bit samples, clipped if it is too loud
    std::vector<short> samples() const
    {
        std::vector<short> out;
        out.reserve(track_.size());
        for (double v : track_) {
            out.push_back(static_cast<short>(std::max(-1.0, std::min(1.0, v)) * 32767));
        }
        return out;
    }

private:
    std::vector<double> track_;
};

/**
 * @class sound_book
 * @brief the music and the effects of the lessons, written out once
 */
class sound_book
{
public:
    /// samples in a step of the music, an eighth note; whole, so that the loop closes exactly
    static constexpr int samples_per_step = 8700;

    sound_book()
    : music_(make_music()),
      fall_(make_fall()),
      treasure_(make_treasure()),
      click_(make_click())
    {}

    /// @return the tune, which loops without a seam
    const std::vector<short> & music() const { return music_; }
    /// @return a slide down and a bump: walking off the cliff
    const std::vector<short> & fall() const { return fall_; }
    /// @return a little fanfare up: reaching the treasure
    const std::vector<short> & treasure() const { return treasure_; }
    /// @return a blip: clicking the top bar
    const std::vector<short> & click() const { return click_; }

    /// @return the melody: 16 bars of 8 steps
    static std::string melody()
    {
        return "E5 - G5 E5 C5 - D5 E5   D5 - G4 B4 D5 - - .     C5 - E5 C5 A4 - B4 C5   A4 - C5 F5 E5 - D5 . "
               "E5 - G5 E5 C6 - B5 G5   B5 A5 G5 D5 G5 - - .    A5 - G5 F5 G5 F5 E5 D5  C5 E5 G5 C6 - - . . "
               "A5 - C6 A5 G5 - F5 .    G5 - E5 G5 C5 - - .     F5 - A5 F5 D5 - E5 F5   G5 - B5 G5 D5 - - . "
               "G5 - E5 B4 E5 - G5 .    A5 - E5 C5 A4 - C5 E5   F5 E5 D5 C5 A4 - C5 F5  G4 B4 D5 F5 G5 - - .";
    }

    /// @return the broken chords under it
    static std::string harmony()
    {
        return "C4 E4 G4 E4 C4 E4 G4 E4  B3 D4 G4 D4 B3 D4 G4 D4  C4 E4 A4 E4 C4 E4 A4 E4  C4 F4 A4 F4 C4 F4 A4 F4 "
               "C4 E4 G4 E4 C4 E4 G4 E4  B3 D4 G4 D4 B3 D4 G4 D4  C4 F4 A4 F4 B3 D4 G4 D4  C4 E4 G4 E4 C4 E4 G4 E4 "
               "C4 F4 A4 F4 C4 F4 A4 F4  C4 E4 G4 E4 C4 E4 G4 E4  D4 F4 A4 F4 D4 F4 A4 F4  B3 D4 G4 D4 B3 D4 G4 D4 "
               "B3 E4 G4 E4 B3 E4 G4 E4  C4 E4 A4 E4 C4 E4 A4 E4  C4 F4 A4 F4 C4 F4 A4 F4  B3 D4 G4 D4 B3 D4 G4 D4";
    }

    /// @return the bass
    static std::string bass()
    {
        return "C3 . C4 . G3 . C4 .  G2 . G3 . D3 . G3 .  A2 . A3 . E3 . A3 .  F2 . F3 . C3 . F3 . "
               "C3 . C4 . G3 . C4 .  G2 . G3 . D3 . G3 .  F2 . F3 . G2 . G3 .  C3 . C4 . G3 . C4 . "
               "F2 . F3 . C3 . F3 .  C3 . C4 . G3 . C4 .  D3 . D4 . A3 . D4 .  G2 . G3 . D3 . G3 . "
               "E3 . E4 . B3 . E4 .  A2 . A3 . E3 . A3 .  F2 . F3 . C3 . F3 .  G2 . G3 . D3 . G3 .";
    }

    /// @return the drums: the pitch sets how bright the noise is
    static std::string drums()
    {
        std::string out;
        for (int bar = 0; bar < 16; bar++) {
            out += "C9 . C8 C9 C9 . C8 . ";
        }
        return out;
    }

private:
    static std::vector<short> make_music()
    {
        const score lead(melody());
        const double step = static_cast<double>(samples_per_step) / sample_rate;
        mixer mix(lead.steps() * samples_per_step);
        pulse_wave half(0.5);
        pulse_wave thin(0.125);
        triangle_wave soft;
        noise_wave hiss;
        envelope plucked;
        plucked.decay = 0.06;
        plucked.sustain = 0.25;
        envelope hit;
        hit.decay = 0.025;
        hit.sustain = 0;
        mix.play(half, lead, step, envelope(), 0.62);
        mix.play(thin, score(harmony()), step, plucked, 0.2);
        mix.play(soft, score(bass()), step, plucked, 0.42);
        mix.play(hiss, score(drums()), step, hit, 0.2);
        return mix.samples();
    }

    static std::vector<short> make_fall()
    {
        mixer mix(static_cast<std::size_t>(0.62 * sample_rate));
        pulse_wave half(0.5);
        noise_wave hiss;
        envelope held;
        held.decay = 0.3;
        held.sustain = 0.7;
        envelope bump;
        bump.decay = 0.05;
        bump.sustain = 0;
        mix.slide(half, 900, 90, 0, 0.5, held, 0.9);
        mix.slide(hiss, 1000, 1000, 0.42, 0.2, bump, 0.9);
        return mix.samples();
    }

    static std::vector<short> make_treasure()
    {
        mixer mix(static_cast<std::size_t>(0.56 * sample_rate));
        pulse_wave quarter(0.25);
        const double hop = 0.07;
        const char * const notes[4] = {"C6", "E6", "G6", "C7"};
        for (int i = 0; i < 4; i++) {
            const double seconds = i == 3 ? 0.35 : hop;
            mix.slide(quarter, frequency_of(notes[i]), frequency_of(notes[i]), i * hop, seconds, envelope(), 0.9);
        }
        return mix.samples();
    }

    static std::vector<short> make_click()
    {
        mixer mix(static_cast<std::size_t>(0.03 * sample_rate));
        pulse_wave half(0.5);
        mix.slide(half, 1200, 1200, 0, 0.03, envelope(), 0.7);
        return mix.samples();
    }

    std::vector<short> music_;
    std::vector<short> fall_;
    std::vector<short> treasure_;
    std::vector<short> click_;
};

/**
 * @class loop_cursor
 * @brief hands out a track piece by piece, going round for ever
 */
class loop_cursor
{
public:
    /// @param track is what to play; it must outlive the cursor
    explicit loop_cursor(const std::vector<short> & track)
    : track_(&track)
    {}

    /// @brief write the next @param frames samples to @param out
    void fill(short * out,
              std::size_t frames)
    {
        for (std::size_t i = 0; i < frames; i++) {
            out[i] = (*track_)[place_];
            place_ = (place_ + 1) % track_->size();
        }
    }

private:
    const std::vector<short> * track_;
    std::size_t place_ = 0;
};
} // namespace lessons
} // namespace games
#endif
