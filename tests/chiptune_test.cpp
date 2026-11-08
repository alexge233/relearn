/**
 * Tests of the lessons' music and sound effects. Nobody can listen in a test, so they measure the samples: the pitch
 * of a note, the shape of each wave, that nothing clips, and that the loop closes on silence.
 */
#include "../examples/window/chiptune.hpp"
#include "../examples/window/lesson.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

namespace lessons = games::lessons;
using lessons::sample_rate;
using lessons::sound_book;

namespace
{
/// @return the upward crossings of zero in samples [first, last), which is the number of cycles of a plain tone
std::size_t rising(const std::vector<short> & s,
                   std::size_t first,
                   std::size_t last)
{
    std::size_t count = 0;
    for (std::size_t i = first + 1; i < last; i++) {
        count += s[i - 1] < 0 && s[i] >= 0;
    }
    return count;
}

/// @return the strength of @param hz in samples [first, last)
double strength(const std::vector<short> & s,
                std::size_t first,
                std::size_t last,
                double hz)
{
    double re = 0;
    double im = 0;
    const double turn = 2 * std::acos(-1.0) * hz / sample_rate;
    for (std::size_t i = first; i < last; i++) {
        re += s[i] * std::cos(turn * i);
        im += s[i] * std::sin(turn * i);
    }
    return std::sqrt(re * re + im * im) / static_cast<double>(last - first);
}

int peak(const std::vector<short> & s)
{
    int most = 0;
    for (short v : s) {
        most = std::max(most, std::abs(static_cast<int>(v)));
    }
    return most;
}

double mean(const std::vector<short> & s)
{
    double sum = 0;
    for (short v : s) {
        sum += v;
    }
    return sum / static_cast<double>(s.size());
}

/// @return a plain tone through a mixer, with a flat envelope
std::vector<short> tone(lessons::waveform & voice,
                        double hz,
                        double seconds = 1)
{
    lessons::mixer mix(static_cast<std::size_t>(seconds * sample_rate));
    lessons::envelope flat;
    flat.sustain = 1;
    mix.slide(voice, hz, hz, 0, seconds, flat, 1);
    return mix.samples();
}
} // namespace

TEST_CASE("note names become frequencies", "[chiptune]")
{
    REQUIRE(lessons::frequency_of("A4") == Approx(440));
    REQUIRE(lessons::frequency_of("A5") == Approx(880));
    REQUIRE(lessons::frequency_of("A3") == Approx(220));
    REQUIRE(lessons::frequency_of("C4") == Approx(261.626).epsilon(0.0001));
    REQUIRE(lessons::frequency_of("B3") == Approx(246.942).epsilon(0.0001));
    REQUIRE(lessons::frequency_of("E5") == Approx(659.255).epsilon(0.0001));
    REQUIRE(lessons::frequency_of("C#4") == Approx(277.183).epsilon(0.0001));
    REQUIRE(lessons::frequency_of("Db4") == Approx(lessons::frequency_of("C#4")));
    REQUIRE(lessons::frequency_of("Bb3") == Approx(233.082).epsilon(0.0001));
}

TEST_CASE("what is not a note is refused", "[chiptune]")
{
    for (const char * bad : {"", "H4", "A", "C#", "A#x", "C##4", "c4", "Ax4"}) {
        REQUIRE_THROWS_AS(lessons::frequency_of(bad), std::invalid_argument);
    }
}

TEST_CASE("a score reads notes, holds and rests", "[chiptune]")
{
    const lessons::score part("A4 - - . C4 D4 -  .");
    REQUIRE(part.steps() == 8);
    REQUIRE(part.notes().size() == 3);
    REQUIRE(part.notes()[0].frequency == Approx(440));
    REQUIRE(part.notes()[0].start == 0);
    REQUIRE(part.notes()[0].length == 3);
    REQUIRE(part.notes()[1].start == 4);
    REQUIRE(part.notes()[1].length == 1);
    REQUIRE(part.notes()[2].start == 5);
    REQUIRE(part.notes()[2].length == 2);
}

TEST_CASE("a hold after a rest stays silent", "[chiptune]")
{
    const lessons::score part("A4 . - - C4");
    REQUIRE(part.steps() == 5);
    REQUIRE(part.notes().size() == 2);
    REQUIRE(part.notes()[0].length == 1);
    REQUIRE(part.notes()[1].start == 4);
    REQUIRE(lessons::score("- . -").notes().empty());
}

TEST_CASE("a pulse wave spends its duty high and averages zero", "[chiptune]")
{
    for (double duty : {0.5, 0.25, 0.125}) {
        lessons::pulse_wave w(duty);
        int high = 0;
        double sum = 0;
        const int n = 8000;
        for (int i = 0; i < n; i++) {
            const double v = w.at(static_cast<double>(i) / n);
            high += v > 0;
            sum += v;
        }
        REQUIRE(static_cast<double>(high) / n == Approx(duty));
        REQUIRE(sum / n == Approx(0).margin(1e-9));
    }
    lessons::pulse_wave quarter(0.25);
    REQUIRE(quarter.at(0.1) == Approx(0.75));
    REQUIRE(quarter.at(0.6) == Approx(-0.25));
}

TEST_CASE("a triangle wave starts at zero and peaks at a quarter", "[chiptune]")
{
    lessons::triangle_wave w;
    REQUIRE(w.at(0) == Approx(0).margin(1e-12));
    REQUIRE(w.at(0.125) == Approx(0.5));
    REQUIRE(w.at(0.25) == Approx(1));
    REQUIRE(w.at(0.5) == Approx(0).margin(1e-12));
    REQUIRE(w.at(0.75) == Approx(-1));
    REQUIRE(w.at(0.875) == Approx(-0.5));
}

TEST_CASE("noise changes once a cycle, and starts the same way each note", "[chiptune]")
{
    lessons::noise_wave w;
    REQUIRE(w.at(0.1) == 0.5);
    REQUIRE(w.at(0.2) == 0.5);
    std::vector<double> first;
    int changes = 0;
    int ones = 0;
    double last = 0.5;
    w.restart();
    w.at(0.5);
    for (int cycle = 0; cycle < 3000; cycle++) {
        const double v = w.at(0.1);
        w.at(0.9);
        first.push_back(v);
        changes += v != last;
        ones += v > 0;
        last = v;
    }
    REQUIRE(changes > 1000);
    REQUIRE(ones > 1200);
    REQUIRE(ones < 1800);
    w.restart();
    w.at(0.5);
    for (double expected : first) {
        REQUIRE(w.at(0.1) == expected);
        w.at(0.9);
    }
}

TEST_CASE("the loudness of a note", "[chiptune]")
{
    lessons::envelope e;
    e.attack = 0.01;
    e.decay = 0.1;
    e.sustain = 0.5;
    e.release = 0.02;
    REQUIRE(e.level(-0.001, 1) == 0);
    REQUIRE(e.level(0, 1) == 0);
    REQUIRE(e.level(0.005, 1) == Approx(0.5));
    REQUIRE(e.level(0.01, 1) == Approx(1));
    REQUIRE(e.level(0.11, 1) == Approx(0.5 + 0.5 * std::exp(-1.0)));
    REQUIRE(e.level(0.9, 1) == Approx(0.5).margin(0.001));
    REQUIRE(e.level(0.99, 1) == Approx(0.25).margin(0.001));
    REQUIRE(e.level(1, 1) == 0);
    REQUIRE(e.level(2, 1) == 0);
}

TEST_CASE("a note sounds at its pitch and in its place", "[chiptune]")
{
    lessons::pulse_wave w(0.5);
    REQUIRE(rising(tone(w, 440), 0, sample_rate) == Approx(440).margin(1));
    lessons::triangle_wave t;
    REQUIRE(rising(tone(t, 110), 0, sample_rate) == Approx(110).margin(1));
    lessons::mixer mix(sample_rate);
    mix.slide(w, 440, 440, 0.25, 0.25, lessons::envelope(), 1);
    const auto s = mix.samples();
    REQUIRE(peak({s.begin(), s.begin() + sample_rate / 4}) == 0);
    REQUIRE(peak({s.begin() + sample_rate / 2 + 1, s.end()}) == 0);
    REQUIRE(peak({s.begin() + sample_rate / 4, s.begin() + sample_rate / 2}) > 15000);
}

TEST_CASE("every note of noise starts the same way", "[chiptune]")
{
    lessons::noise_wave hiss;
    lessons::envelope flat;
    flat.sustain = 1;
    lessons::mixer mix(2000);
    mix.slide(hiss, 4000, 4000, 0, 0.02, flat, 1);
    mix.slide(hiss, 4000, 4000, 0.025, 0.02, flat, 1);
    const auto s = mix.samples();
    const auto gap = static_cast<std::size_t>(0.025 * sample_rate);
    for (std::size_t i = 0; i < static_cast<std::size_t>(0.02 * sample_rate); i++) {
        REQUIRE(s[i] == s[gap + i]);
    }
}

TEST_CASE("a held note sounds for all its steps", "[chiptune]")
{
    const double step = 0.1;
    lessons::mixer mix(static_cast<std::size_t>(0.5 * sample_rate));
    lessons::pulse_wave w(0.5);
    lessons::envelope flat;
    flat.sustain = 1;
    mix.play(w, lessons::score("A4 - - . ."), step, flat, 1);
    const auto s = mix.samples();
    const auto at = [](double seconds) { return static_cast<std::size_t>(seconds * sample_rate); };
    REQUIRE(peak({s.begin() + at(0.2), s.begin() + at(0.29)}) > 15000);
    REQUIRE(peak({s.begin() + at(0.3) + 1, s.end()}) == 0);
}

TEST_CASE("the level scales a note, and voices add up", "[chiptune]")
{
    lessons::envelope flat;
    flat.sustain = 1;
    lessons::pulse_wave w(0.5);
    lessons::mixer one(4410);
    one.slide(w, 440, 440, 0, 0.1, flat, 0.5);
    lessons::mixer two(4410);
    two.slide(w, 440, 440, 0, 0.1, flat, 0.25);
    two.slide(w, 440, 440, 0, 0.1, flat, 0.25);
    REQUIRE(peak(one.samples()) == peak(two.samples()));
    REQUIRE(peak(one.samples()) == Approx(0.25 * 32767).margin(100));
}

TEST_CASE("what is too loud is clipped, never wrapped round", "[chiptune]")
{
    lessons::envelope flat;
    flat.sustain = 1;
    lessons::pulse_wave w(0.5);
    lessons::mixer mix(4410);
    mix.slide(w, 440, 440, 0, 0.1, flat, 8);
    const auto s = mix.samples();
    REQUIRE(peak(s) == 32767);
    REQUIRE(s[1000] * s[1001] >= 0);
    std::size_t flips = 0;
    for (std::size_t i = 1; i < s.size(); i++) {
        flips += (s[i - 1] > 0) != (s[i] > 0);
    }
    REQUIRE(flips < 100);
}

TEST_CASE("a note past the end of the track is cut, not written out of it", "[chiptune]")
{
    lessons::pulse_wave w(0.5);
    lessons::mixer mix(100);
    mix.slide(w, 440, 440, 0.001, 1, lessons::envelope(), 1);
    REQUIRE(mix.samples().size() == 100);
}

TEST_CASE("all the voices of the tune last the same", "[chiptune][music]")
{
    const std::size_t steps = 16 * 8;
    REQUIRE(lessons::score(sound_book::melody()).steps() == steps);
    REQUIRE(lessons::score(sound_book::harmony()).steps() == steps);
    REQUIRE(lessons::score(sound_book::bass()).steps() == steps);
    REQUIRE(lessons::score(sound_book::drums()).steps() == steps);
}

TEST_CASE("the tune loops without a seam and does not clip", "[chiptune][music]")
{
    const sound_book book;
    const auto & music = book.music();
    REQUIRE(music.size() == 16 * 8 * sound_book::samples_per_step);
    REQUIRE(std::abs(static_cast<int>(music.front())) < 50);
    REQUIRE(std::abs(static_cast<int>(music.back())) < 50);
    REQUIRE(peak(music) < 32767);
    REQUIRE(peak(music) > 20000);
    REQUIRE(std::fabs(mean(music)) < 500);
}

TEST_CASE("the tune plays the melody it is written with", "[chiptune][music]")
{
    const sound_book book;
    const auto & music = book.music();
    const std::size_t step = sound_book::samples_per_step;
    // the first note is E5, for two steps; a G#5 is in no chord of the tune
    REQUIRE(strength(music, 0, 2 * step, lessons::frequency_of("E5")) > 5 * strength(music, 0, 2 * step, lessons::frequency_of("G#5")));
    // the third step is G5
    REQUIRE(strength(music, 2 * step, 3 * step, lessons::frequency_of("G5")) > 3 * strength(music, 2 * step, 3 * step, lessons::frequency_of("E5")));
    // every part is in the mix: the bass opens on C3, and its second bar on G2
    REQUIRE(strength(music, 0, step, lessons::frequency_of("C3")) > 3 * strength(music, 0, step, lessons::frequency_of("D#3")));
    REQUIRE(strength(music, 8 * step, 9 * step, lessons::frequency_of("G2")) > 3 * strength(music, 8 * step, 9 * step, lessons::frequency_of("C#3")));
}

TEST_CASE("every voice is in the mix, at its own loudness", "[chiptune][music]")
{
    const sound_book book;
    const auto & music = book.music();
    const double step = static_cast<double>(sound_book::samples_per_step) / sample_rate;
    // how much of a voice, rendered alone, is in the tune: 1 if it is all there, 0 if it is missing
    const auto share = [&](lessons::waveform & voice, const std::string & text, const lessons::envelope & shape, double level) {
        lessons::mixer alone(music.size());
        alone.play(voice, lessons::score(text), step, shape, level);
        const auto part = alone.samples();
        double common = 0;
        double own = 0;
        for (std::size_t i = 0; i < music.size(); i++) {
            common += static_cast<double>(music[i]) * part[i];
            own += static_cast<double>(part[i]) * part[i];
        }
        return common / own;
    };
    lessons::envelope plucked;
    plucked.decay = 0.06;
    plucked.sustain = 0.25;
    lessons::envelope hit;
    hit.decay = 0.025;
    hit.sustain = 0;
    lessons::pulse_wave half(0.5);
    lessons::pulse_wave thin(0.125);
    lessons::triangle_wave soft;
    lessons::noise_wave hiss;
    REQUIRE(share(half, sound_book::melody(), lessons::envelope(), 0.62) == Approx(1).margin(0.1));
    // the quiet harmony shares notes with the loud melody, which adds about a half; alone it would be 0.56
    REQUIRE(share(thin, sound_book::harmony(), plucked, 0.2) == Approx(1.56).margin(0.2));
    REQUIRE(share(soft, sound_book::bass(), plucked, 0.42) == Approx(1).margin(0.1));
    REQUIRE(share(hiss, sound_book::drums(), hit, 0.2) == Approx(1).margin(0.1));
}

TEST_CASE("every bar starts on a note of its chord", "[chiptune][music]")
{
    const lessons::score melody(sound_book::melody());
    const lessons::score harmony(sound_book::harmony());
    const auto same_pitch_class = [](double a, double b) {
        const double semitones = 12 * std::log2(a / b);
        return std::fabs(semitones / 12 - std::round(semitones / 12)) < 0.01;
    };
    for (std::size_t bar = 0; bar < 16; bar++) {
        double first = 0;
        for (const auto & n : melody.notes()) {
            if (n.start == bar * 8) {
                first = n.frequency;
            }
        }
        REQUIRE(first > 0);
        bool fits = false;
        for (const auto & n : harmony.notes()) {
            if (n.start >= bar * 8 && n.start < bar * 8 + 4) {
                fits = fits || same_pitch_class(first, n.frequency);
            }
        }
        REQUIRE(fits);
    }
}

TEST_CASE("the effects say what happened", "[chiptune][effects]")
{
    const sound_book book;
    const auto & fall = book.fall();
    const auto & treasure = book.treasure();
    // a fall slides down: the crossings thin out
    const std::size_t third = static_cast<std::size_t>(0.5 * sample_rate) / 3;
    REQUIRE(rising(fall, 0, third) > 3 * rising(fall, 2 * third, 3 * third));
    // the treasure climbs: each of its notes is higher
    const std::size_t hop = static_cast<std::size_t>(0.07 * sample_rate);
    REQUIRE(rising(treasure, 0, hop) < rising(treasure, hop, 2 * hop));
    REQUIRE(rising(treasure, hop, 2 * hop) < rising(treasure, 2 * hop, 3 * hop));
    REQUIRE(rising(treasure, 2 * hop, 3 * hop) < rising(treasure, 3 * hop, 4 * hop));
    REQUIRE(book.click().size() < static_cast<std::size_t>(sample_rate) / 20);
    for (const auto * s : {&fall, &treasure, &book.click()}) {
        REQUIRE(std::abs(static_cast<int>(s->front())) < 50);
        REQUIRE(std::abs(static_cast<int>(s->back())) < 300);
        REQUIRE(peak(*s) < 32767);
        REQUIRE(peak(*s) > 8000);
    }
}

TEST_CASE("the sounds are the same every time", "[chiptune]")
{
    const sound_book one;
    const sound_book two;
    REQUIRE(one.music() == two.music());
    REQUIRE(one.fall() == two.fall());
}

TEST_CASE("a cursor goes round a track for ever, in pieces of any size", "[chiptune]")
{
    const std::vector<short> track = {1, 2, 3, 4, 5};
    lessons::loop_cursor cursor(track);
    short out[13];
    cursor.fill(out, 3);
    REQUIRE(out[0] == 1);
    REQUIRE(out[2] == 3);
    cursor.fill(out, 13);
    const short expected[13] = {4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1};
    for (int i = 0; i < 13; i++) {
        REQUIRE(out[i] == expected[i]);
    }
    cursor.fill(out, 1);
    REQUIRE(out[0] == 2);
}

TEST_CASE("a step's sound", "[chiptune][cue]")
{
    using lessons::cue;
    REQUIRE(lessons::cue_of(false, false) == cue::none);
    REQUIRE(lessons::cue_of(true, false) == cue::fall);
    REQUIRE(lessons::cue_of(false, true) == cue::treasure);
    REQUIRE(lessons::cue_of(true, true) == cue::fall);
}

TEST_CASE("steps only make a sound when they can be followed", "[chiptune][cue]")
{
    lessons::controls c;
    while (c.rate() > 1) {
        c.slower();
    }
    while (c.rate() <= 10) {
        REQUIRE(c.followable_steps());
        c.faster();
    }
    REQUIRE(!c.followable_steps());
    c.paused = true;
    REQUIRE(c.followable_steps());
}

TEST_CASE("the sound has its key, its bubble and its option", "[chiptune]")
{
    using lessons::command;
    REQUIRE(lessons::command_of_key('M') == command::sound);
    REQUIRE(lessons::menu_help(command::sound).find("(M)") != std::string::npos);
    REQUIRE(lessons::controls().sound);
    const char * none[] = {"p"};
    REQUIRE(!lessons::lesson_options::parse(1, const_cast<char **>(none)).mute);
    const char * muted[] = {"p", "--mute"};
    REQUIRE(lessons::lesson_options::parse(2, const_cast<char **>(muted)).mute);
}
