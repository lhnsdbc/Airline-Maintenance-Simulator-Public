from config import G

class AcSubtype:
    ''' Class for aricraft Subtypes'''
    def __init__(self,
                 id,
                 IATA,
                 detail_code,
                 name,
                 DT,        # Time for departure preparation
                 TAT,       # Turn around time
                 AT         # Time for preparing aircraft after arrival
                 ):
        self.id = id
        self.IATA = IATA
        self.detail_code = detail_code
        self.name = name
        self.DT = DT
        self.TAT = TAT
        self.AT = AT

        self.aircraft = []



class Aircraft:
    '''Class for aircraft registrations'''
    def __init__(
            self,
            id,                 # Registration
            subtype,
            asia_tail,           # carrier-specific subtype flag retained for schema compatibility
            state='on ground',
            coordinates=None
    ):
        self.id = id
        self.subtype = subtype
        self.asia = asia_tail
        self.state = state
        self.coordinates = coordinates

        self.rotations = []
        self.rotations_executed = []
        self.reserve_slots = []
        self.slots = []
        self.slots_LM = []
        self.slots_executed = []
        # self.flights = []
        self.tasks_open = []
        self.tasks_executed = []
        self.tasks_missed = []
        self.duty_next = None
        self.duty_last = None
        self.duty_current = None
        self.process = None
        self.expected_ready_time_AMS = None
        self.grounded = False
        self.type = None
        # Cumulative usage proxy (PAPER_DESIGN NR prediction); fleet-average elapsed estimate
        self.cum_fh = 0.0
        self.cum_fc = 0.0


    def update_coordinates_flying(self):
        if self.state!='flying':
            raise Exception('This function can only be called if aircraft is flying')
        self.coordinates = self.flights[-1].next_waypoint()

    # UNUSED: empty stub, never implemented or called (legacy from old repo).
    def compute_expected_time_AMS(self):
        pass
