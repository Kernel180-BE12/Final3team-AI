# 알림톡 블랙리스트

{% hint style="info" %}
알림톡으로 발송이 불가 템플릿 유형을 안내드립니다.

* 알림톡 블랙리스트는 정보성 메시지 여부와 관계없이 알림톡으로 발송 할 수 없는 메시지에 해당됩니다.
* 템플릿 승인 후, 모니터링을 통해 가이드 위반이 확인될 경우 해당 템플릿에 대해 차단 조치가 취해질 수 있습니다.
{% endhint %}



(1) 포인트 지급에 명시적으로 동의하지 않은 수신자에게 발송하는 포인트 적립/소멸 메시지(마일리지, 쿠폰, 적립금 포함) 발송

* <mark style="color:red;">포인트 적립.소멸 메시지 발송을 위해서는 거래 및 계약상 내용에 지급되는 포인트의 발급 조건, 지급되는 포인트의 금액 등을 명시적으로 기재 후 수신자에게 동의를 받아야 합니다.</mark>

{% hint style="danger" %}
안녕하세요! OOO 입니다.\
\#{고객명} 님의 소멸 예정 적립금을 안내 드립니다.  \
\
■소멸 예정 적립금\
\- 소멸 예정 적립금 : #{10,000}원\
\- 유효 기간 : #{00월 00일 24시 소멸}\
\
유효 기간이 지나면 자동으로 소멸되어 사용이 불가능합니다\
<mark style="background-color:red;">※ 이 메시지는 이용약관(계약서) 동의에 따라 지급된 포인트의 소멸 안내 메시지입니다.</mark>
{% endhint %}

(2) 쿠폰 발급(마일리지, 포인트, 적립금 포함) 후 빠른 시일 내 소멸하는 쿠폰의 메시지

{% hint style="danger" %}
안녕하세요 OO님 \
쿠폰 사용 종료일이 얼마 남지 않았습니다. \
\
<mark style="background-color:red;">\[쿠폰 정보 안내]</mark> \ <mark style="background-color:red;">쿠폰명 : 전상품 5,000원 할인 쿠폰</mark> \ <mark style="background-color:red;">만료일 : 오늘 단 하루 ⏰</mark>\
\
※ 쿠폰 지급에 동의한 회원에게 발송하는 안내 메시지입니다.\
※유효기간이 지나면 사용 여부와 관계없이 자동 소멸되어 사용이 불가능합니다.
{% endhint %}

(3) 무료 뉴스레터 등 무료 구독형 메시지 불가 (친구톡 사용 권장)

{% hint style="danger" %}
<mark style="background-color:red;">\[OOOO 뉴스레터]  구독신청하신 <언론사람> 뉴스레터 #{해당월}월호가 발행되었습니다.</mark>\ <mark style="background-color:red;"> 콘텐츠 바로보기: #{LINK}</mark>
{% endhint %}

(4) 계약관계이며 반드시 전달되어야 하는 메시지 외의 공지 메시지 불가 (친구톡 사용 권장)

{% hint style="danger" %}
(사)OOO복지회의 봉사자님들 <mark style="background-color:red;">2016년 한 해 동안 수고하셨습니다.</mark>
{% endhint %}

(5) 변수만으로 구성된 메시지 불가 (ex. #{상품명} #{송장정보})

{% hint style="danger" %}
<mark style="background-color:red;">#{club\_name} #{book\_date} #{book\_time} #{appl\_name}님 #{phone} #{amt} #{memo}</mark>
{% endhint %}

(6) 장바구니 상품 삭제 예정 안내 메시지 불가

{% hint style="danger" %}
안녕하세요. #{NAME}님 혹시 <mark style="background-color:red;">장바구니에 담긴상품 잊지않으셨나요?</mark>&#x20;

<mark style="background-color:red;">꼭 확인해주세요!</mark> #{PRODUCT\_LIST}
{% endhint %}

(7) 특가 상품 알림 메시지 불가

{% hint style="danger" %}
OO GOLF <mark style="background-color:red;">특가알리미 #{월/일(요일)} 1박2일#{지역} #{상품명}#{간단포함사항}#{상품가} 진행 희망 시 연락 바랍니다.</mark>
{% endhint %}

(8) 생일 축하 메시지 불가

{% hint style="danger" %}
\[OO병원 건강증진센터] #{이름}님, <mark style="background-color:red;">생일을 축하드립니다.</mark>
{% endhint %}

(9) 설문작성 요청 시 자체 포인트 또는 쿠폰 제공 안내 (단, 별도의 상품 가능)

{% hint style="danger" %}
\#{계약자명}님 OOO 멤버스 상품을 구매해주셔서 감사드립니다. 1초 설문하고 <mark style="background-color:red;">포인트 받아가세요!</mark> \
설문조사 바로가기  #{url}
{% endhint %}

(10) 금융사고 예방 목적으로 결제/송금/납부 유도 메시지 불가

* 템플릿 본문 내, ‘결제하세요’, ‘납부하세요’, ‘송금하세요’ 등의 안내 멘트 포함 가능&#x20;
* 템플릿 본문 내, 결제/송금/납부 수단 언급 불가능(무통장 입금 등 결제/송금/납부 방법은 언급 가능)&#x20;
* 템플릿 버튼 내, ‘결제’, ‘납부’, ‘송금’, ‘청구서’, ‘청구’를 포함하여 이용자의 액션을 유도하는 멘트 언급 불가능&#x20;
* 템플릿 본문 내 링크 또는 버튼을 통하여 연결 된 랜딩 페이지에서 결제/송금/납부 기능 제공 불가능

단, 다음의 내용이 확인되는 업체에 한하여 가능(결제사 확인되는 결제 페이지 캡쳐 첨부하여 심사 요청)\
&#x20; \- [금융위원회](https://www.fcsc.kr/B/fu_b_06.jsp)에 선불전자지급수단발행업, 전자지급결제대행업(PG), 결제대금예치업(ESCROW) 중 2개 이상 등록이 확인된 업체 \
&#x20; \- 휴대폰 제조사 결제대행업 업체 중 제1  금융권의 시중은행 소속의 카드사 연동이 5개 이상 적용된 업체

{% hint style="info" %}
**\[결제/송금/납부 유도 '가능'메시지 가이드]**

·템플릿 본문 내, ‘결제하세요’, ‘납부하세요’, ‘송금하세요’ 등의 안내 멘트 포함 가능

·템플릿 본문 내, 결제/송금/납부 수단 언급 가능

·템플릿 버튼 내, ‘결제’, ‘납부’, ‘송금’, ‘청구서’, ‘청구’를 포함하여 이용자의 액션을 유도하는 멘트 언급 가능

·템플릿 본문 내 링크 또는 버튼을 통하여 연결 된 랜딩 페이지에서 결제/송금/납부 기능 제공 가능
{% endhint %}

{% hint style="danger" %}
\[OOO 렌트카]&#x20;

\#{고객명} 고객님의 렌트카 예약이 가능합니다. \
온라인 예약금을 6시간 내에 결제해 주시기 바랍니다. \
<mark style="background-color:red;">OOO페이를 이용하여 간편하게 결제 가능하며</mark> 미결제 시 고객님의 예약은 자동으로 취소됩니다.&#x20;

※렌트카 온라인 예약금은 나의 예약확인에서 확인 가능합니다.\


버튼: <mark style="background-color:red;">결제하러 가기</mark>
{% endhint %}

(11) 장바구니 등록 상품 안내

{% hint style="danger" %}
<mark style="background-color:red;">\[#{고객명}]님 장바구니에 담은 뒤 #{3일}이상 구매하지 않은 상품이 #{4개} 있습니다.</mark> \ <mark style="background-color:red;">잊기전에 구매하세요.☞ #{url}</mark>
{% endhint %}

(12) 쇼핑몰에서 클릭했던 상품 안내

{% hint style="danger" %}
안녕하세요#{고객명}님, \
<mark style="background-color:red;">#{고객명}님께서 클릭하셨던 상품을 한번 더 보여드릴께요. ^^</mark>\ <mark style="background-color:red;">링크를 통해서 바로 확인하세요! #{url}</mark>
{% endhint %}

(13) 앱 다운로드 안내 및 URL

{% hint style="danger" %}
더 자세한 사항은 앱을 통해 확인 할 수 있습니다. \
<mark style="background-color:red;">다운받기 :</mark> [<mark style="background-color:red;">https://appsto.re/kr/\_ooooo</mark>](https://appsto.re/kr/_ooooo)
{% endhint %}

(14) 수술/진료/검사 후 안부 문자

{% hint style="danger" %}
\#{성명}고객님 #{병원명}입니다. \
어느덧 수술 하신지 3개월이 경과 되었습니다. <mark style="background-color:red;">수술하신 부위는 좀 어떠신지요?</mark> \
<mark style="background-color:red;">궁금한사항 있으시거나 경과예약시 언제든 연락 주세요^^</mark>&#x20;



\[OO성형외과] \
경과예약: 02-123-4567
{% endhint %}

(15) 전화통화 후 수신자의 요청 없이 자동으로 발송하는 안내 문자

{% hint style="danger" %}
<mark style="background-color:red;">\[#{업체명}]</mark>\ <mark style="background-color:red;">전화주셔서 감사합니다. 유선상 문의하신 매장정보를 안내해 드립니다.</mark>\ <mark style="background-color:red;">\*매장전화: #{매장번호}</mark>\ <mark style="background-color:red;">\*상점 정보 보기: #{URL}</mark>
{% endhint %}

(16) 카카오톡 채널 추가 후, 발송되는 카카오톡 채널 추가 확인 메시지

{% hint style="danger" %}
<mark style="background-color:red;">OOO의 카카오톡 채널을 추가해 주셔서 감사합니다.</mark>\ <mark style="background-color:red;">앞으로 다양한 이벤트 소식과 유용한 정보를 메세지로 받으실 수 있습니다.</mark>\ <mark style="background-color:red;">있는 그대로 아름다운 나를 사랑하는 또 하나의 방법 \_ OOO</mark>
{% endhint %}

(17) 카카오톡 채널 추가 후, 2년 주기로 발송되는 수신동의 확인 메시지(일반적인 광고수신 동의 확인은 가능)

{% hint style="danger" %}
<mark style="background-color:red;">\[정기적 수신동의 확인 안내]</mark>\ <mark style="background-color:red;">안녕하세요, OOO입니다.</mark>\ <mark style="background-color:red;">이 메시지는 정보통신망법에 따라 2년마다 발송되는 광고성 정보 수신동의 확인메시지 입니다.</mark>



<mark style="background-color:red;">이전 수신동의일자 :</mark>\ <mark style="background-color:red;">앞으로도 유익한 소식과 다양한 혜택으로 찾아 뵙겠습니다.</mark>\ <mark style="background-color:red;">메시지 수신을 원치 않으시면 "홈>친구차단" 버튼을 눌러주시거나 내 카카오톡 채널 목록에서 차단해주세요. 감사합니다.</mark>
{% endhint %}

(18) 비 제도권금융회사에서 발송하는 주식 종목 추천 메시지

{% hint style="danger" %}
<mark style="background-color:red;">급등주가 도착했습니다◀</mark>

<mark style="background-color:red;">무료체험 당첨을 축하드립니다.</mark>\ <mark style="background-color:red;">대한민국 상위 1% 전문가의 VIP전용 추천종목을 무료로 받아보실 수 있습니다.</mark>\ <mark style="background-color:red;">서비스 이용방법은 아래 버튼을 통해서 확인해보세요!</mark>

<mark style="background-color:red;">▼ 아래 버튼 클릭 ▼</mark>\ <mark style="background-color:red;">무료리딩방 입장</mark> [<mark style="background-color:red;">https://open.kakao.com/</mark>](https://open.kakao.com/)<mark style="background-color:red;">OOO으로 찾아 뵙겠습니다.</mark>\


<mark style="background-color:red;">메시지 수신을 원치 않으시면 "홈>친구차단" 버튼을 눌러주시거나 내 카카오톡 채널 목록에서 차단해주세요. 감사합니다.</mark>
{% endhint %}

(19) 비광고성 메시지에 재화 판매 등 광고성 정보가 연결된 URL을 포함하는 메시지

{% hint style="danger" %}
<mark style="background-color:red;">\[주문완료 안내] #{홈쇼핑} 주문완료 #{주문내역} #{주문내역 확인}</mark>

<mark style="background-color:red;"> 이달의 특가 상품 바로가기 : #{URL}</mark>
{% endhint %}

(20)이용하지 않는 대출의 정보 안내

{% hint style="danger" %}
<mark style="background-color:red;">#{고객명}님, 조회하셨던 00대출 조건이 달라졌어요.</mark>&#x20;

&#x20; <mark style="background-color:red;">변경된 대출 조건을 확인해보세요</mark>
{% endhint %}

(21)신용정보 조회/변경 메시지에 광고 목적을 포함하는 메시지

{% hint style="danger" %}
<mark style="background-color:red;">고객님을  위한  다양한 대출상품 비교가 준비 되어있어요.</mark>

\#{기관명}에서 내 신용정보를 조회했어요.조회 사유를 확인해보세요.&#x20;

사유  확인은 신용 점수에 영향을 주지 않아요.
{% endhint %}

(22)연령 인증이 필요하여 발송이  불가한 업종 및 콘텐츠의 메시지 불가\
(단, 성인 인증 채널은 [채널 메시지 집행 가이드](https://kakaobusiness.gitbook.io/main/ad/moment/start/messagead/guide#undefined-10)에 안내된 예외 발송이 허용되는 업종 및 콘텐츠의 메시지  발송이 가능)

* “청소년 이용 불가 등급의 게임물" 관련 메시지
* “주류 등 청소년 유해 관련 콘텐츠”를 포함한 메시지
* "주류 무상 이벤트및  시음 제공"을 포함한 메시지

{% hint style="danger" %}
이 메시지는 \[OOOOO]사전예약을 신청하신 사용자에게 발송되는 메시지 입니다.

\[OOOOO] 정식 출시! 사전예약 신청 감사드리며 특별한 아이템을 받을 수 있는 쿠폰 번호를 드립니다.\


 다운로드 바로가기: #{다운로드URL}

 쿠폰 번호: #{쿠폰번호}

 공식카페 바로가기

\#{url}
{% endhint %}