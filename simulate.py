'''
Polling places

Val Alvern Cueco Ligo

Main file for polling place simulation
'''

import sys
import random
import queue
import click
import util


class Voter:
    '''
    Initializes the Voter class

    Methods:
        start: Sets the start_time attribute
        depart: Sets the departure_time attribute
        vote: Sets the has_voted attribute

    Attributes:
        arrival_time: (int) the time the voter arrives at the polls
        voting_duration: (int) the amount of time the voter takes to vote
        is_impatient: (bool) whether the voter is impatient
        start_time: (int) the time the voter is assigned to a voting booth
        departure_time: (int) the time the voter has left the voting booth
        has_voted: (bool) whether or not they voted
    '''

    def __init__(self, arrival_time, voting_duration, is_impatient):
        '''
        Initialize the voters

        Attributes:
        arrival_time: (int) the time the voter arrives at the polls
        voting_duration: (int) the amount of time the voter takes to vote
        is_impatient: (bool) whether the voter is impatient
        start_time: (int) the time the voter is assigned to a voting booth
        departure_time: (int) the time the voter has left the voting booth
        has_voted: (bool) whether or not they voted

        '''
        self.arrival_time = arrival_time
        self.voting_duration = voting_duration
        self.is_impatient = is_impatient
        self.start_time = None
        self.departure_time = None
        self.has_voted = False

    def start(self, start_time):
        '''
        Sets the voters start time
        '''

        self.start_time = start_time

    def depart(self, departure_time):
        '''
        Sets the voters departure_time
        '''

        self.departure_time = departure_time

    def vote(self, has_voted):
        '''
        Sets the has_voted bool flag
        '''

        self.has_voted = has_voted

    def __str__(self):
        '''
        the voter:
            arrives at some time, voted or not, is patient or not,
            the voting duration, the departure time, etc.
        '''
        return 'Voter arrived at' + self.arrival_time\
               + 'and starting voting at' + self.start_time +\
               'and ended voting at' + self.departure_time + '.' +\
               'The voter took' + self.voting_duration + 'to vote.'

class VotingBooths:
    '''Class for representing a bank of voting booths.

    Attributes: None

    Methods:
        is_booth_available: bool
            is there at least one unoccupied booth
        is_some_booth_occupied: bool
            is there at least one occupied booth
        enter_booth(v):
            add a voter to a booth. requires a booth to be available.
        time_next_free(): float
            when will a booth be free next (only called when all the
            booths are occupied)
        exit_booth():
             remove the next voter to depart from the booths and
             return the voter and their departure_time.
    '''

    def __init__(self, num_booths):
        '''
        Initialize the voting booths.

        Args:
            num_booths: (int) the number of voting booths in the bank
        '''

        self._num_booths = num_booths
        self._q = queue.PriorityQueue()

    def is_booth_available(self):
        '''Is at least one booth open'''
        return self._q.qsize() < self._num_booths

    def is_some_booth_occupied(self):
        '''Is at least one booth occupied'''
        return self._q.qsize() > 0

    def enter_booth(self, v):
        '''
        Add voter v to an open booth

        Args:
            v: (Voter) the voter to add to the booth.

        Requirements: there must be an open booth.
        '''
        assert self.is_booth_available(), "All booths in use"
        assert v.start_time, "Voter's start time must be set."

        dt = v.start_time + v.voting_duration
        self._q.put((dt, v))

    def time_next_free(self):
        '''
        When will the next voter leave?

        Returns: next departure time

        Requirements: there must be at least one occupied booth.
        '''
        assert self.is_some_booth_occupied(), "No booths in use"

        # PriorityQueue does not have a peek method.
        # So, do a get followed by a put.
        (dt, v) = self._q.get()
        self._q.put((dt, v))
        return dt


    def exit_booth(self):
        '''
        Remove voter with lowest departure time.

        Returns: the voter and the voter's departure time

        Requirements: there must be at least one occupied booth.
        '''
        assert self.is_some_booth_occupied(), "No booths in use"

        (dt, v) = self._q.get()
        return v, dt


class Precinct(object):
    '''
    Class for representing precincts.

    Attributes: None

    Methods:
        simulate(seed, voting_booths, impatience_threshold):
            Simulate election day for the precinct using the
            specified seed, voting_booths, and impatience threshold.
    '''

    def __init__(self, name, hours_open, num_voters, arrival_rate,
                 voting_duration_rate, impatience_prob):
        '''
        Constructor for the Precinct class

        Input:
            name: (str) Name of the precinct
            hours_open: (int) Hours the precinct will remain open
            num_voters: (int) Number of voters in the precinct
            arrival_rate: (float) Rate at which voters arrive
            voting_duration_rate: (float) Lambda for voting duration
            impatience_prob: (float) the probability that a voter
                will be impatient.
        '''

        self.name = name
        self.hours_open = hours_open
        self.num_voters = num_voters
        self.arrival_rate = arrival_rate
        self.voting_duration_rate = voting_duration_rate
        self.impatience_prob = impatience_prob

    def __voterlist(self, seed):
        '''
        Generate list of voters according to given parameters

        Args:
            seed: (int) the seed for the random number generator

        Returns:
            List of registered voters who arrived in time
        '''
        random.seed(seed)
        voters = []
        time = 0
        close_time = self.hours_open * 60
        for _ in range(self.num_voters):
            gap, voting_duration, is_impatient = \
                util.gen_voter_parameters(self.arrival_rate,
                                          self.voting_duration_rate,
                                          self.impatience_prob)
            arrival_time = time + gap
            time = arrival_time
            if arrival_time > close_time:
                break
            voter = Voter(arrival_time, voting_duration, is_impatient)
            voters.append(voter)

        return voters

    def simulate(self, seed, voting_booths, impatience_threshold):
        '''
        Simulate election day for the precinct using the specified seed,
        voting_booths, and impatience threshold.

        Args:
            seed: (int) the seed for the random number generator
            voting_booths: (VotingBooths) the voting booths assigned to the
                precinct for the day
            impatience_threshold: (int) the number of minutes an impatient voter
                is willing to wait (inclusive)

        Returns: list of Voters
        '''

        voters = self.__voterlist(seed)
        for voter in voters:
            if voting_booths.is_booth_available():
                voter.start(voter.arrival_time)
                voting_booths.enter_booth(voter)
                voter.depart(voter.start_time + voter.voting_duration)

            else:
                next_time = voting_booths.time_next_free()
                if voter.arrival_time > next_time:
                    v, _ = voting_booths.exit_booth()
                    v.vote(True)
                    voter.start(voter.arrival_time)
                    voting_booths.enter_booth(voter)
                    voter.depart(voter.start_time + voter.voting_duration)
                else:
                    if voter.is_impatient and \
                            next_time - voter.arrival_time > \
                            impatience_threshold:
                        continue
                    else:
                        voter.start(next_time)
                        v, _ = voting_booths.exit_booth()
                        v.vote(True)
                        voting_booths.enter_booth(voter)
                        voter.depart(voter.start_time + voter.voting_duration)

        while voting_booths.is_some_booth_occupied():
            v, _ = voting_booths.exit_booth()
            v.vote(True)

        return voters


def impatience_simulate(seed, precinct, voting_booths):
    '''
    For a given precinct, simulate individual impatience threshold

    Args:
        seed (int): the initial seed for the random number generator
        precinct: (Precinct) the precinct to simulate
        voting_booths: (VotingBooths) the voting booths assigned to the
                precinct for the day

    Return:
        (int) the impatience threshold that results in all voters voting
    '''

    threshold = 1
    while True:
        voters = precinct.simulate(seed, voting_booths, threshold)
        voted_ls = [v.has_voted for v in voters]
        if False in voted_ls:
            threshold += 10
        else:
            break

    return threshold

def find_impatience_threshold(seed, precinct, num_booths, num_trials):
    '''
    For a given precinct, find the impatience threshold at which all
    voters are likely to vote.

    Args:
        seed (int): the initial seed for the random number generator
        precinct: (Precinct) the precinct to simulate
        num_booths: (int) number of voting booths to use in
            the simulations
        num_trials: (int) the number of trials to run

    Returns: (int) the median threshold from the trials

    '''

    assert num_trials > 0

    epochs = []
    for _ in range(num_trials):
        imp = impatience_simulate(seed,
                precinct, VotingBooths(num_booths))
        epochs.append(imp)
        seed = seed + 1

    impatience = sorted(epochs)[len(epochs) // 2]

    return impatience

def voting_booths_simulate(seed, precinct, imp_threshold):
    '''
    For a given precinct, generate number of voting booths which results in

    Args:
        seed (int): the initial seed for the random number generator
        precinct: (Precinct) the precinct to simulate
        imp_threshold: (int) the impatience threshold

    Returns:
        (int) the number of booths that result in all voters voting
    '''

    num_booths = 1

    while True:
        voters = precinct.simulate(seed,
                VotingBooths(num_booths), imp_threshold)
        voted_ls = [v.has_voted for v in voters]
        if False in voted_ls:
            num_booths += 1
        else:
            break

    return num_booths

def find_voting_booths_needed(seed, precinct, imp_threshold, num_trials):
    '''
    For a given precinct, seed, and impatience threshold, predict the number of
    booths needed to make it likely that all the voters will vote.

    Args:
        seed (int): the initial seed for the random number generator
        precinct: (Precinct) the precinct to simulate
        imp_threshold: (int) the impatience threshold
        num_trials: (int) the number of trials to run

    Returns: (int) the median number of booths needed from the trials.
    '''

    assert num_trials > 0

    epochs = []
    for _ in range(num_trials):
        impatience = voting_booths_simulate(seed, precinct, imp_threshold)
        epochs.append(impatience)
        seed = seed + 1

    med = sorted(epochs)[len(epochs) // 2]

    return med


@click.command(name="simulate")
@click.argument('precinct_file', type=click.Path(exists=True))
@click.option('--num-booths', type=int, default=1,
              help="number of voting booths to use")
@click.option('--impatience-threshold', type=float,
              default=1000, help="the impatience threshold")
@click.option('--print-voters', is_flag=True)
@click.option('--find-threshold', is_flag=True)
@click.option('--find-num-booths', is_flag=True)
@click.option('--num-trials', type=int, default=100,
              help="number trials to run")
def cmd(precinct_file, num_booths, impatience_threshold,
        print_voters, find_threshold, find_num_booths, num_trials):
    '''
    Run the program...
    '''
    #pylint: disable=too-many-locals
    p, seed = util.load_precinct(precinct_file)

    precinct = Precinct(p["name"],
                        p["hours_open"],
                        p["num_voters"],
                        p["arrival_rate"],
                        p["voting_duration_rate"],
                        p["impatience_prob"])

    if find_threshold:
        pt = find_impatience_threshold(seed, precinct, num_booths, num_trials)
        s = ("Given {} booths, an impatience threshold of {}"
             " would be appropriate for Precinct {}")
        print(s.format(num_booths, pt, p["name"]))
    elif find_num_booths:
        vbn = find_voting_booths_needed(seed, precinct,
                                        impatience_threshold, num_trials)
        s = ("Given an impatience threshold of {}, provisioning {}"
             " booth(s) would be appropriate for Precinct {}")
        print(s.format(impatience_threshold, vbn, p["name"]))
    elif print_voters:
        vb = VotingBooths(num_booths)
        voters = precinct.simulate(seed, vb, impatience_threshold)
        util.print_voters(voters)
    else:
        vb = VotingBooths(num_booths)
        voters = precinct.simulate(seed, vb, impatience_threshold)
        print("Precinct", p["name"])
        print("- {} voters voted".format(len(voters)))
        if len(voters) > 0:
            # last person might be impatient.  look
            # backwards for the first actual voter.
            last_voter_departure_time = None
            for v in voters[::-1]:
                if v.departure_time:
                    last_voter_departure_time = v.departure_time
                    break
            s = "- Polls closed at {} and last voter departed at {:.2f}."
            print(s.format(p["hours_open"]*60, last_voter_departure_time))
            nv = len([v for v in voters if not v.has_voted])
            print("- {} voters left without voting".format(nv))
            if not voters[-1].departure_time:
                print("  including the last person to arrive at the polls")


if __name__ == "__main__":
    cmd() # pylint: disable=no-value-for-parameter
