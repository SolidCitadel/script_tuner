# 정성 샘플 — formal → casual spoken 변환

`t5gemma2-1b-SBCSAE-lora-es`(조기종료판) 모델의 test split 예측 일부.

- **입력(formal)**: 파이프라인 `pairs` 단계에서 LLM이 생성한 문어체 문장
- **출력(prediction)**: 본 모델이 변환한 구어체
- 표기: `<pause:short>` / `<pause:long>`은 발화 휴지(쉼) 마커
- ⚠️ 참조(reference, SBCSAE 원문 발화)는 라이선스(CC BY-ND)상 본 문서에서 생략.
  분포 비교는 [`metrics_summary.md`](metrics_summary.md) 참조.

---

**1.**
> **입력:** We will not test them. So I had Bill come over with his Leatherman tool, or whatever it is. A few minutes later he had it undone.
>
> **출력:** `<pause:short>` We won't test them . `<pause:short>` So I had Bill come over with his Leatherman tool or whatever it is . `<pause:long>` A few minutes later he had it undone .

**2.**
> **입력:** There's a big bottle of it down in that bottle rack thing. That's already open and I drank out of it, so you can just give me that.
>
> **출력:** There's a big bottle of it down in that bottle rack thing . That's already open and I drank out of it so . `<pause:short>` You can just give me that .

**3.**
> **입력:** And they come up to me, Miss Flinn, Miss Flinn, and I'd walk over by the rules, I'd point to number two, and I'd just look at them.
>
> **출력:** and they come up to me miss flinn miss flinn and I'd `<pause:long>` walk over `<pause:long>` by the rules and I'd point to number two and I'd just look at em .

**4.**
> **입력:** They're right by the gate. Everyone- even one or two of them. I said I'll advise if the left becomes available, and they said...
>
> **출력:** they're right by the gate . `<pause:long>` Everyone even one or two of em I said I'll advise if the left becomes available and they said ...

---

관찰: 격식체의 "will not"→"won't", "them"→"em" 같은 축약, 휴지 마커 삽입,
대문자·구두점 완화 등 구어 특징이 반영된다. 4번처럼 입력이 끊긴 경우 출력도
끊긴 채로 따라가는 경향이 있다.
