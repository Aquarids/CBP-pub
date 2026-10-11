"""Synthetic contract fixtures only, never scientific study outcomes."""
from fractions import Fraction as Q
import unittest
from cbp.bounded_score import *

ROSTER=tuple(str(i) for i in range(1,11))

def witness(eta='0.5'):
    b=IdentitySetBounds(ROSTER,4,eta)
    for t in range(2):
        rows=[]
        for a in ROSTER:
            r={j:'5' for j in ROSTER if j!=a}
            if a=='1':
                r.update({str(j):('4.4' if (j<=6)==(t==0) else '5.6') for j in range(3,11)})
            rows.append(ScoreRow(a,str(t),f'{a}-{t}',r))
        b.add_rows(rows)
    return b

class Checks(unittest.TestCase):
    def test_strict_joint_tightening(self):
        b=witness()
        self.assertEqual(set(b.intervals),{('1','2','3','4','5','6'),('1','2','7','8','9','10')})
        self.assertEqual(b.count_bounds('1',5),(1,1))
        self.assertEqual(b.taskwise_count_bounds('1',5),(0,2))
    def test_exact_touching(self):
        b=witness('0.6')
        self.assertEqual(len(b.intervals),210)
        self.assertEqual(b.count_bounds('1',5),(1,2))
    def test_common_shift(self):
        b=IdentitySetBounds(ROSTER,4,'0.5')
        b.add_rows([ScoreRow('1',str(t),str(t),{j:5 for j in ROSTER if j!='1'}) for t in range(4)])
        self.assertEqual(b.count_bounds('1',5),(0,4))
        self.assertEqual(len(b.intervals),210)
    def test_missing_invalid_never_dropped(self):
        b=IdentitySetBounds(ROSTER,4,'0.5')
        b.add_rows([ScoreRow('1','t','a',{'2':'NaN','3':11,'4':True,'5':5.0})])
        self.assertEqual(len(b.intervals),210)
        self.assertEqual(b.count_bounds('1',5),(0,1))
    def test_contradiction_fails_closed(self):
        b=IdentitySetBounds(ROSTER,4,0)
        b.add_rows([ScoreRow('1','t','a',{str(j):j for j in range(2,11)})])
        self.assertTrue(b.model_inconsistent)
        self.assertIsNone(b.count_bounds('1',5))
    def test_atomic_validation_and_immutability(self):
        b=IdentitySetBounds(ROSTER,4,0); r={'2':5}
        b.add_rows([ScoreRow('1','t','a',r)]); r['2']=0
        self.assertEqual(b.rows[0].reports['2'],5)
        with self.assertRaises(ValueError): b.add_rows([ScoreRow('1','t','b',{})])
        self.assertEqual(len(b.rows),1)
    def test_actual_honest_set_survives_all_fault_counts(self):
        for f in range(5):
            b=IdentitySetBounds(ROSTER,4,'0.5'); honest=ROSTER[f:]
            rows=[]
            for a in ROSTER:
                r={j:('6.75' if int(j)%2 else '7.25') if j in honest else '0' for j in ROSTER if j!=a}
                rows.append(ScoreRow(a,'t',a,r))
            b.add_rows(rows)
            self.assertIn(tuple(honest[:6]),b.intervals)
            for a in ROSTER:
                lo,hi=b.count_bounds(a,7)
                self.assertLessEqual(lo,1); self.assertGreaterEqual(hi,1)
    def test_census_ambiguity_and_cost(self):
        for aggregation in ('joint_counts','joint_hulls','taskwise','median_interval'):
            m=BoundedScoreMarket(candidates=[{'agent_id':a,'quote':1} for a in ROSTER],task_ids=['x','y'],task_category='fixture',F=4,looks=(1,2),aggregation=aggregation)
            while not m.stopped:
                tasks=m.next_round()
                m.record_round({a:{'task_id':t,'artifact_id':f'{a}-{t}','reports':{j:5 for j in ROSTER if j!=a},'deadline_passed':True} for a,t in tasks.items()})
            self.assertEqual(m.costs,{'rounds':2,'probe_executions':20,'review_requests':180})
            self.assertTrue(all(p['status']=='budget_ambiguity' for p in m.publications().values()))
            self.assertTrue(all(p['point_grade']=='C' for p in m.publications().values()))
    def test_threshold_equality_is_pass(self):
        b=IdentitySetBounds(ROSTER,4,0)
        b.add_rows([ScoreRow('1','t','a',{j:5 for j in ROSTER if j!='1'})])
        self.assertEqual(b.count_bounds('1',5),(1,1))
    def test_cross_producer_elimination_revises_old_bounds(self):
        b=IdentitySetBounds(ROSTER,4,'0.5')
        b.add_rows([ScoreRow('1','t','a',{'2':'5',**{str(j):'4.4' if j<=6 else '5.6' for j in range(3,11)}})])
        self.assertEqual(b.count_bounds('1',5),(0,1))
        b.add_rows([ScoreRow('2','t','b',{j:4 if int(j)<=6 else 6 for j in ROSTER if j!='2'})])
        self.assertEqual(b.count_bounds('1',5),(0,0))
    def test_tolerant_publication_is_not_strict_admission(self):
        m=BoundedScoreMarket(candidates=[{'agent_id':a,'quote':1} for a in ROSTER],task_ids=[str(t) for t in range(10)],task_category='fixture',F=4,looks=(10,),eta=0)
        while not m.stopped:
            tasks=m.next_round()
            m.record_round({a:{'task_id':t,'artifact_id':f'{a}-{t}','reports':{j:9 if int(t)<4 else 7 for j in ROSTER if j!=a},'deadline_passed':True} for a,t in tasks.items()})
        self.assertEqual(m.certified_grade('1'),'A')
        self.assertNotIn('A',m.candidate_list()[0]['certified_levels'])
        self.assertIn('A',m.tolerance_candidate_list()[0]['tolerance_qualified_levels'])
        m.summaries['1']['A']['lower']=Q(3,10)
        m.summaries['1']['A']['upper']=Q(6,10)
        self.assertIsNone(m.certified_grade('1'))
    def test_duplicate_pairs_rejected(self):
        b=IdentitySetBounds(ROSTER,4,0)
        with self.assertRaises(ValueError):b.add_rows([ScoreRow('1','t','a',[('2',5),('2',6)])])

    def test_cached_counts_match_full_rescan(self):
        b=witness('0.6')
        for a in ROSTER:
            indices=[i for i,r in enumerate(b.rows) if r.producer==a]
            for c in (Q(5),Q(7),Q(9),Q(49,10)):
                expected=(min(sum(v[i][0]>=c for i in indices) for v in b.intervals.values()),max(sum(v[i][1]>=c for i in indices) for v in b.intervals.values()))
                self.assertEqual(b.count_bounds(a,c),expected)
                self.assertEqual(b.count_bounds(a,c),expected)
        self.assertEqual(b.taskwise_count_bounds('1',5),b.taskwise_count_bounds('1',5))

    def test_market_inconsistency_keeps_exact_serializable_bounds(self):
        import json
        m=BoundedScoreMarket(candidates=[{'agent_id':a,'quote':1} for a in ROSTER],task_ids=['t'],task_category='fixture',F=4,looks=(1,),eta=0)
        m.record_round({a:{'task_id':t,'artifact_id':a,'reports':{j:int(j) for j in ROSTER if j!=a},'deadline_passed':True} for a,t in m.next_round().items()})
        self.assertTrue(m.bounds.model_inconsistent)
        self.assertEqual(m.candidate_list(),[])
        self.assertEqual(m.tolerance_candidate_list(),[])
        for p in m.publications().values():
            self.assertIsNone(p['certified_grade'])
            self.assertEqual(p['status'],'model_inconsistent')
            for bounds in p['cumulative_bounds'].values():
                self.assertIsInstance(bounds['lower'],Q)
                self.assertIsInstance(bounds['upper'],Q)
                self.assertEqual((exact(bounds['lower']),exact(bounds['upper'])),(0,1))
        json.dumps(m.publications(),default=str)

    def test_baselines_do_not_borrow_global_alarm(self):
        # Each row has an exact six-identity explanation, but the explanations
        # switch; no single size-six set explains both. Outside-bound fixture.
        for aggregation in ('joint_counts','joint_hulls','taskwise','median_interval'):
            m=BoundedScoreMarket(candidates=[{'agent_id':a,'quote':1} for a in ROSTER],task_ids=['x','y'],task_category='fixture',F=4,looks=(2,),eta=0,aggregation=aggregation)
            for round_index in range(2):
                tasks=m.next_round()
                H=set(ROSTER[:6] if round_index==0 else ROSTER[4:])
                m.record_round({a:{'task_id':t,'artifact_id':f'{a}-{t}','reports':{j:5 if j in H else 0 for j in ROSTER if j!=a},'deadline_passed':True} for a,t in tasks.items()})
            self.assertTrue(m.bounds.model_inconsistent)
            self.assertEqual(m.model_inconsistent,aggregation.startswith('joint'))
            if not aggregation.startswith('joint'):
                self.assertTrue(all(p['status']!='model_inconsistent' for p in m.publications().values()))

    def test_sorted_window_exhaustive_equivalence(self):
        from itertools import product
        for m,F in ((4,1),(6,2)):
            roster=tuple(str(i) for i in range(m))
            for eta in ('0','0.25','0.5'):
                b=IdentitySetBounds(roster,F,eta)
                for values in product((None,'0','0.5','1'),repeat=m-1):
                    row=b.normalize(ScoreRow('0','t','a',{j:v for j,v in zip(roster[1:],values) if v is not None}))
                    intervals=[v for H in b.candidates if (v:=b.interval(H,row)) is not None]
                    expected=None if not intervals else (min(v[0] for v in intervals),max(v[1] for v in intervals))
                    self.assertEqual(b.taskwise_interval(row),expected)
    def test_sorted_window_m10_random_exact_equivalence(self):
        import random
        rng=random.Random(1971)
        for eta in ('0','0.25','0.5','0.6','1'):
            b=IdentitySetBounds(ROSTER,4,eta)
            for t in range(100):
                raw={j:rng.choice([None,'invalid','0','4.4','4.5','5','5.5','5.6','10']) for j in ROSTER if j!='1'}
                row=b.normalize(ScoreRow('1',str(t),str(t),raw))
                intervals=[v for H in b.candidates if (v:=b.interval(H,row)) is not None]
                expected=None if not intervals else (min(v[0] for v in intervals),max(v[1] for v in intervals))
                self.assertEqual(b.taskwise_interval(row),expected)

    def test_outside_bound_not_guaranteed(self):
        b=IdentitySetBounds(ROSTER,4,0)
        # Producer10 honest, five faulty reporters can support an incorrect world.
        b.add_rows([ScoreRow('10','t','a',{j:10 if int(j)<=5 else 0 for j in ROSTER if j!='10'})])
        self.assertEqual(b.count_bounds('10',5),(1,1))

if __name__=='__main__':unittest.main()
