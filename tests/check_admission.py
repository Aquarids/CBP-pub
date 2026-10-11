import itertools
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from cbp.admission import AdmissionConfig, AdmissionState, Cell, Review
from cbp.admission.committees import bias_bound, sample_committee
from cbp.admission.controller import ReviewController
from cbp.admission.ledger import Bid, MarketLedger, ResourceLedger, choose_bid, private_bid_policy
from cbp.admission.providers import Artifact, FileReviewer, ProgramReviewer
from cbp.admission.smoke import run_smoke


class AdmissionChecks(unittest.TestCase):
    def state(self, cap=128, bias=0):
        cell = Cell('a', 'v1', 'skill')
        return cell, AdmissionState((cell,), frozenset(['x', 'y', 'z']), AdmissionConfig(max_probes_per_cell=cap,bias_floor=bias))

    def reviews(self, score=1):
        return [Review(x,score,'fixture') for x in ['x','y','z']]

    def test_anytime_formula_and_budget(self):
        c,s=self.state();self.assertEqual(s.summary(c)['lower'],0)
        s.observe(c,'1',self.reviews())
        self.assertAlmostEqual(s.summary(c)['radius'],math.sqrt(math.log(4/.05)/2))
        self.assertAlmostEqual(sum(1/(m*(m+1)) for m in range(1,10000)),1-1/10000)

    def test_committee_failure_exhaustive(self):
        for n in range(1,8):
            for f in range(n+1):
                for k in range(1,n+1,2):
                    groups=list(itertools.combinations(range(n),k))
                    exact=sum(sum(x<f for x in g)>k//2 for g in groups)/len(groups)
                    self.assertAlmostEqual(bias_bound(n,f,k,0,0)['median_failure_bound'],exact)
        self.assertEqual(bias_bound(5,3,5,0,0)['bias_floor'],1)
        self.assertEqual(bias_bound(5,2,5,.1,0)['bias_floor'],.1)
        self.assertNotIn('a',sample_committee(['a','b','c','d'],'a',3,random.Random(1)))

    def test_validation(self):
        for value in [float('nan'),float('inf'),-1,True]:
            with self.assertRaises(ValueError):Review('x',value,'fixture')
        for kwargs in [{'delta':0},{'delta':1},{'committee_size':2},{'max_probes_per_cell':False}]:
            with self.assertRaises(ValueError):AdmissionConfig(**kwargs)
        c,s=self.state()
        with self.assertRaises(ValueError):s.observe(c,'1',[Review('a',1,'fixture')]+self.reviews()[:2])
        with self.assertRaises(ValueError):s.observe(c,'1',[self.reviews()[0]]*3)
        s.observe(c,'1',self.reviews())
        with self.assertRaises(ValueError):s.observe(c,'1',self.reviews())

    def test_cap_and_irreducible_floor(self):
        c,s=self.state(cap=2,bias=.7)
        for i in range(2):s.observe(c,str(i),self.reviews())
        self.assertEqual(s.summary(c)['status'],'inconclusive')
        with self.assertRaises(ValueError):s.observe(c,'3',self.reviews())

    def test_pending_idempotence_and_timeout(self):
        c,s=self.state();controller=ReviewController(s)
        controller.begin(c,'1','artifact',['x','y','z'])
        controller.submit(c,'1','artifact',self.reviews()[0])
        self.assertEqual(s.summary(c)['n'],0)
        with self.assertRaises(ValueError):controller.begin(c,'2','new',['x','y','z'])
        with self.assertRaises(ValueError):controller.submit(c,'1','changed',self.reviews()[1])
        controller.submit(c,'1','artifact',self.reviews()[1]);controller.submit(c,'1','artifact',self.reviews()[2])
        controller.submit(c,'1','artifact',self.reviews()[2]);self.assertEqual(s.summary(c)['n'],1)
        controller.begin(c,'2','next',['x','y','z']);self.assertEqual(controller.stop(c,'2'),'inconclusive')
        self.assertEqual(s.summary(c)['status'],'inconclusive')
        self.assertIsNone(s.rank('skill')['winner'])
        with self.assertRaises(ValueError):controller.begin(c,'3','next',['x','y','z'])

    def test_human_file_and_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            human=FileReviewer(tmp,'x');ticket=human.submit('r1','task','artifact')
            self.assertIsNone(human.poll(ticket))
            Path(tmp,ticket+'.response.json').write_text(json.dumps({'review_id':'r1','reviewer_id':'x','score':.8,'source':'human'}))
            self.assertEqual(human.poll(ticket).score,.8)
            with self.assertRaises(ValueError):human.submit('r1','task','changed')
            with self.assertRaises(ValueError):human.poll('../invalid')
        self.assertEqual(ProgramReviewer('x',lambda t,a:.8).review('task','artifact').score,.8)

    def test_private_bids_and_cash_resource_separation(self):
        self.assertIsNone(private_bid_policy('a',2,.1,1))
        bid=private_bid_policy('a',.2,.1,1)
        self.assertAlmostEqual(bid.amount,.3);self.assertFalse(hasattr(bid,'modeled_execution_cost'))
        self.assertIsNone(choose_bid([bid],{'a':1},0,1))
        self.assertIsNone(choose_bid([bid],{'a':1},1,.1))
        money=MarketLedger(1,{'a':0});money.settle_formal('task',bid)
        self.assertAlmostEqual(money.user_balance+sum(money.provider_balances.values()),1)
        self.assertEqual(money.summary()['peer_review_transfers'],0)
        with self.assertRaises(ValueError):money.settle_formal('task',bid)
        resource=ResourceLedger();resource.record('a','probe',Artifact('a','fixture',1))
        self.assertIsNone(resource.summary()['external_monetary_cost'])
        self.assertEqual(resource.summary()['modeled_resource_units'],1)

    def test_smoke_is_deterministic_and_labeled(self):
        a=run_smoke();self.assertEqual(a,run_smoke());self.assertEqual(a['selected'],'strong')
        self.assertEqual(a['estimates']['weak']['status'],'rejected')
        self.assertEqual(a['market_ledger']['peer_review_transfers'],0)
        self.assertGreater(a['resource_metrics']['modeled_resource_units'],0)
        self.assertIsNone(a['resource_metrics']['tokens'])
        self.assertIsNone(run_smoke(1)['selected'])

    def test_currency_is_not_silently_combined(self):
        ledger=ResourceLedger()
        with self.assertRaises(ValueError):Artifact('a','fixture',1,monetary_cost=1)
        ledger.record('a','probe',Artifact('a','fixture',1,monetary_cost=1,currency='USD'))
        ledger.record('a','review',Artifact('a','fixture',1,monetary_cost=2,currency='CNY'))
        self.assertIsNone(ledger.summary()['external_monetary_cost'])
        self.assertEqual(ledger.summary()['known_cost_subtotals_by_currency'],{'USD':1,'CNY':2})

    def test_accuracy_floor_and_scheduler(self):
        c,s=self.state(bias=.2)
        self.assertEqual(s.accuracy(c)['decision'],'resolution_floor')
        self.assertIsNone(s.next_cell('skill','accuracy'))
        self.assertEqual(s.next_cell('skill'),c)
        c,s=self.state(cap=1)
        s.observe(c,'1',self.reviews())
        self.assertEqual(s.accuracy(c)['decision'],'inconclusive')
        self.assertIsNone(s.next_cell('skill'))

    def test_ranking_zero_and_equality_boundary(self):
        from unittest.mock import patch
        a,b=Cell('a','v1','s'),Cell('b','v1','s')
        s=AdmissionState((a,b),frozenset(['x','y','z']),AdmissionConfig(epsilon=.1))
        self.assertEqual(s.rank('s')['status'],'pending')
        values={a:{'n':1,'lower':.5,'upper':.9},b:{'n':1,'lower':.2,'upper':.6}}
        with patch.object(s,'summary',side_effect=lambda c: values[c]):
            result=s.rank('s')
            self.assertEqual(result['status'],'epsilon_resolved')
            self.assertEqual(result['winning_cell']['configuration_id'],'v1')
        values[b]['upper']=.600001
        with patch.object(s,'summary',side_effect=lambda c: values[c]):
            self.assertEqual(s.rank('s')['status'],'pending')

    def test_all_terminal_reasons_override_decisions(self):
        for reason in ['timeout','withdrawn','assumption_unavailable']:
            for completed in [False,True]:
                c,s=self.state()
                if completed:s.observe(c,'old',self.reviews())
                controller=ReviewController(s)
                controller.begin(c,'new','artifact',['x','y','z'])
                controller.stop(c,'new',reason)
                self.assertEqual(s.summary(c)['status'],'inconclusive' if reason=='timeout' else reason)
                self.assertEqual(s.rank('skill')['status'],'inconclusive')
                self.assertIsNone(s.next_cell('skill'))
                with self.assertRaises(ValueError):s.observe(c,'bypass',self.reviews())

    def test_ranking_scheduler_capped_leaders_do_not_stall(self):
        cells=tuple(Cell(x,'v1','s') for x in ['a','b','c'])
        state=AdmissionState(cells,frozenset(['x','y','z']),AdmissionConfig(max_probes_per_cell=25))
        for cell,score,count in zip(cells,[1,.5,0],[25,25,24]):
            for i in range(count):state.observe(cell,str(i),self.reviews(score))
        self.assertEqual(state.rank('s')['status'],'pending')
        self.assertEqual(state.next_cell('s','ranking'),cells[2])
        state.observe(cells[2],'24',self.reviews(0))
        self.assertEqual(state.rank('s')['status'],'inconclusive')
